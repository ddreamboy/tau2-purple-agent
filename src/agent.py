import json
import logging
import os
import re
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional customer service agent for an airline. Your job is to help users with flight bookings, modifications, cancellations, baggage, refunds, and compensation.

---

## BEHAVIORAL RULES (Priority Order — follow top rules over lower ones)

### 1. TOOL USAGE (Highest Priority)
- NEVER simulate, pretend, or invent results of actions — ALWAYS call the actual tool
- Make ONLY ONE tool call per response — never chain tools in a single reply
- ALWAYS retrieve data with a lookup tool before any write/modify action
- If a tool returns an error, tell the user clearly and offer alternatives

### 2. IDENTITY VERIFICATION
- You MUST obtain the user's `user_id` before calling ANY tool
- If user hasn't provided it, ask once clearly before proceeding

### 3. CONFIRMATION PROTOCOL (for all destructive actions)
Before booking, canceling, modifying a flight, or updating baggage:
  Step 1 — Call the relevant lookup tool and show the user the retrieved details
  Step 2 — State EXACTLY what action you are about to take
  Step 3 — Ask: "Can you confirm you'd like to proceed?" — then WAIT for explicit "yes"
  Do NOT proceed until confirmation is received.

### 4. INFORMATION POLICY
- Only share information retrieved from tools or provided by the user
- Never invent flight details, prices, policies, or passenger data

---

## TOOL CALL FORMAT

When you need to call a tool, respond with ONLY this JSON — no prose, no explanation:

{"tool_calls":[{"id": "call_1", "name": "tool_name", "arguments": {"arg1": "value1"}}]}

When you receive tool results, they arrive as:
{"tool_results":[{"id": "call_1", "result": {...}}]}

After receiving tool results:
- If the result is an error → inform the user and offer alternatives
- If the result is data → use it to continue the task or present to user
- If another tool call is needed → make it (one at a time)
- If task is complete → respond to the user in plain text

---

## RESPONSE STYLE
- Be concise and professional — avoid filler phrases
- When showing flight/booking details, use a clear structured format
- If you cannot fulfill a request, explain WHY (referencing policy) and proactively offer what you CAN do
- Never apologize excessively — one acknowledgment is enough
"""


class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_API_URL", "https://routerai.ru/api/v1"),
        )
        self.model = os.environ.get("LLM_API_BASE_MODEL", "google/gemini-2.5-flash-lite")
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _parse_tool_results(self, text: str) -> list[dict] | None:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool_results" in data:
                return data["tool_results"]
        except (json.JSONDecodeError, TypeError):
            pass

        if self.messages and self.messages[-1].get("tool_calls"):
            tc_id = self.messages[-1]["tool_calls"][0].get("id", "call_1")
            try:
                data = json.loads(text)
                return [{"id": tc_id, "result": data}]
            except (json.JSONDecodeError, TypeError):
                return [{"id": tc_id, "result": {"output": text}}]

        return None

    def _parse_tool_calls_response(self, text: str) -> list[dict] | None:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                clean = match.group(0)
                data = json.loads(clean)
                if "tool_calls" in data:
                    return data["tool_calls"]
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        text = get_message_text(message)

        tool_results = self._parse_tool_results(text)
        if tool_results:
            for tr in tool_results:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("id", "call_1"),
                        "content": json.dumps(tr.get("result", {})),
                    }
                )
            logger.info(f"Received {len(tool_results)} tool results")
        else:
            self.messages.append({"role": "user", "content": text})
            logger.info(f"User message: {text[:200]}")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            await updater.complete(
                new_agent_text_message("I'm sorry, I encountred a technical issue. Please try again.")
            )
            return

        choice = response.choices[0]
        reply = choice.message.content or ""

        logger.info(f"Model reply (first 300 chars): {reply[:300]}")

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_calls = choice.message.tool_calls
            self.messages.append(
                {
                    "role": "assistant",
                    "content": reply or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            tc = tool_calls[0]
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            payload = json.dumps({"name": tc.function.name, "arguments": args})
            logger.info(f"Native tool calls mapped: {payload[:300]}")
            await updater.complete(new_agent_text_message(payload))
            return

        parsed_tool_calls = self._parse_tool_calls_response(reply)
        if parsed_tool_calls:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for i, tc in enumerate(parsed_tool_calls)
                    ],
                }
            )

            tc = parsed_tool_calls[0]
            payload = json.dumps({"name": tc["name"], "arguments": tc.get("arguments", {})})
            logger.info(f"JSON tool calls mapped: {payload[:300]}")
            await updater.complete(new_agent_text_message(payload))
            return

        if not reply:
            reply = "I'm sorry, could you please repeat your request?"

        self.messages.append({"role": "assistant", "content": reply})
        logger.info(f"Text response to user: {reply[:200]}")
        await updater.complete(new_agent_text_message(reply))
