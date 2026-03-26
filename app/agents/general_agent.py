from collections.abc import Callable
from typing import Optional

from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult


class GeneralAgent:
    # GeneralAgent is the fallback when the request does not clearly belong to
    # planning, API design, or database modeling.
    name = "general"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(
        self,
        user_message: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResult:
        prompt = (
            "You are the General Agent of a software assistant system.\n"
            "Handle general programming questions, broad explanations, and requests "
            "that do not clearly belong to planning, API design, or database modeling.\n"
            "Reply clearly and directly.\n\n"
            f"User request: {user_message}"
        )

        response = self.llm_client.generate(prompt)

        return AgentResult(
            agent_name=self.name,
            content=response,
            used_tools=[]
        )
