from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.planner.define_user_flows import define_user_flows
from app.tools.planner.extract_core_features import extract_core_features
from app.tools.planner.generate_mvp_plan import generate_mvp_plan


class PlannerAgent:
    name = "planner"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

        self.tools = {
            "generate_mvp_plan": generate_mvp_plan,
            "extract_core_features": extract_core_features,
            "define_user_flows": define_user_flows,
        }

    def choose_tool(self, user_message: str) -> str:
        prompt = (
            "You are the Planner Agent of a software assistant system.\n"
            "Choose the best planning tool for the user's request.\n"
            "Available tools:\n"
            "- generate_mvp_plan: for generating a short MVP plan\n"
            "- extract_core_features: for identifying the main features of the idea\n"
            "- define_user_flows: for describing the main user flows of the system\n\n"
            "Reply with only one tool name.\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        if selected_tool not in self.tools:
            return "generate_mvp_plan"

        return selected_tool

    def handle(self, user_message: str) -> AgentResult:
        selected_tool = self.choose_tool(user_message)
        tool_function = self.tools[selected_tool]
        tool_result = tool_function(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tool=selected_tool
        )