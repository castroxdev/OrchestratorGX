from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.planner.define_user_flows import define_user_flows


class PlannerAgent:
    name = "planner"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(self, user_message: str) -> AgentResult:
        tool_result = define_user_flows(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tool="define_user_flows"
        )