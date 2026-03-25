from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult


class DatabaseAgent:
    name = "database"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(self, user_message: str) -> AgentResult:
        prompt = (
            "You are the Database Agent of a software assistant system.\n"
            "Handle requests related to database design, entities, relationships, "
            "SQL schemas, and data modeling.\n"
            "Reply clearly and directly.\n\n"
            f"User request: {user_message}"
        )

        response = self.llm_client.generate(prompt)

        return AgentResult(
            agent_name=self.name,
            content=response
        )