import logging
import os

from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.utils import get_message_text, new_agent_text_message


class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_API_URL", "https://routerai.ru/api/v1"),
        )
        self.model = os.environ.get("LLM_API_BASE_MODEL", "google/gemini-2.5-flash-lite")
        self.messages = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        text = get_message_text(message)
        self.messages.append({"role": "user", "content": text})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})

        logging.info(f"Agent reply: {reply[:200]}")
        await updater.complete(new_agent_text_message(reply))
