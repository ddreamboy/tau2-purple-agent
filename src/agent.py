import json
import logging
import os
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

## AIRLINE POLICY REFERENCE

### Cancellations — allowed ONLY if:
- Booking was made within the last 24 hours, OR
- The airline cancelled or significantly changed the flight, OR
- Passenger is in Business class, OR
- Passenger has travel insurance AND provides a valid reason (health/weather)
- All other cases: deny and explain clearly

### Cabin Class Changes
- Allowed only if NO flight in the reservation has departed yet
- Must apply the same cabin class to ALL flights in the reservation
- Cannot do partial upgrades

### Baggage
- Extra bag fee: $50 per bag
- Never add bags the user did not explicitly request
- Always confirm bag count before adding

### Travel Insurance
- Cost: $30 per passenger
- MUST be purchased at booking time — cannot be added retroactively
- Benefit: enables full refund for health or weather-related cancellations

### Human Agent Transfer
- When requested by user, OR when policy prevents fulfilling a request
- Action: call `transfer_to_human_agents` tool FIRST, then reply:
  "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
- Never transfer without calling the tool

---

## TOOL CALL FORMAT

When you need to call a tool, respond with ONLY this JSON — no prose, no explanation:

{"tool_calls": [{"id": "call_1", "name": "tool_name", "arguments": {"arg1": "value1"}}]}

When you receive tool results, they arrive as:
{"tool_results": [{"id": "call_1", "result": {...}}]}

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

---

## EXAMPLE INTERACTION FLOW

User: "I want to cancel my flight"
Agent: "I'd be happy to help with that. Could you please provide your user ID?"

User: "My ID is 12345"
Agent: [calls lookup_reservation tool]

After tool result:
Agent: "I found your reservation: [details]. 
Before I proceed — this booking was made 3 days ago and is Economy class, so standard cancellation policy applies. 
Unfortunately, this doesn't qualify for a free cancellation. Would you like me to transfer you to a human agent to discuss options?"
"""


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
