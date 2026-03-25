from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.api.generate_api_endpoints import generate_api_endpoints


class APIAgent:
    name = "api"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(self, user_message: str) -> AgentResult:
        tool_result = generate_api_endpoints(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tool="generate_api_endpoints"
        )