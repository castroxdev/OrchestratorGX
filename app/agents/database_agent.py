from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.database.generate_sql_schema import generate_sql_schema


class DatabaseAgent:
    name = "database"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(self, user_message: str) -> AgentResult:
        tool_result = generate_sql_schema(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tool="generate_sql_schema"
        )