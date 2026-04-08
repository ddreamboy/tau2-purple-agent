import json
import logging
import os
from typing import Any

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful customer service agent for an airline. You help users with booking, modifying, and canceling flight reservations, as well as refunds and compensation.

## Core Rules
- Always use tools to perform actions - NEVER pretend to complete an action without calling the actual tool
- Always get the user_id from the user before calling any tools
- Before any database-modifying action (booking, cancel, modify, baggage update), you MUST:
  1. Retrieve and show the relevant details using a lookup tool
  2. List exactly what you will do
  3. Ask for explicit user confirmation (wait for "yes")
- Only make ONE tool call at a time - never chain multiple tool calls in one response
- Never make up information - always use tools to get real data
- You should not provide information not available through tools or user messages

## Policy Summary
- Cancellations allowed if: booked within last 24h OR airline cancelled flight OR business class OR user has travel insurance with valid reason
- Cabin class changes: allowed if no flight has departed yet, must use same class for all flights in reservation
- Baggage: $50/extra bag, do not add bags user does not need
- Travel insurance: $30/passenger, enables full refund for health/weather reasons, must be added at booking time
- Transfer to human: call transfer_to_human_agents tool THEN say "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
- Deny requests that violate policy

## Tool Call Format
When you need to call a tool, respond with ONLY this JSON (no other text):
{"tool_calls": [{"id": "call_1", "name": "tool_name", "arguments": {"arg1": "value1"}}]}

When you receive tool results, they will be in the next message as:
{"tool_results": [{"id": "call_1", "result": {...}}]}

## Important
- After receiving tool results, continue the task - make another tool call or respond to the user
- If a user request cannot be fulfilled per policy, explain clearly why and offer to transfer to a human agent
- Be concise but helpful in responses to users"""


class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_API_URL", "https://routerai.ru/api/v1"),
        )
        self.model = os.environ.get(
            "LLM_API_BASE_MODEL", "google/gemini-2.5-flash-lite"
        )
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _parse_tool_results(self, text: str) -> list[dict] | None:
        """Try to parse incoming message as tool results."""
        try:
            data = json.loads(text)
            if "tool_results" in data:
                return data["tool_results"]
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _parse_tool_calls_response(self, text: str) -> list[dict] | None:
        """Try to parse model response as tool calls JSON."""
        try:
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
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
                new_agent_text_message(
                    "I'm sorry, I encountred a technical issue. Please try again."
                )
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
            payload = json.dumps(
                {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                        for tc in tool_calls
                    ]
                }
            )
            logger.info(f"Native tool calls: {payload[:300]}")
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
            payload = json.dumps({"tool_calls": parsed_tool_calls})
            logger.info(f"JSON tool calls: {payload[:300]}")
            await updater.complete(new_agent_text_message(payload))
            return

        if not reply:
            reply = "I'm sorry, could you please repeat your request?"

        self.messages.append({"role": "assistant", "content": reply})
        logger.info(f"Text response to user: {reply[:200]}")
        await updater.complete(new_agent_text_message(reply))
