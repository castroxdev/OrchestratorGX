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
            "Your task is to choose the best option for the user's request.\n"
            "Available options:\n"
            "- generate_mvp_plan: use this for requests asking for an MVP plan, phased plan, or minimum viable scope\n"
            "- extract_core_features: use this for requests asking for core features, main features, or essential capabilities\n"
            "- define_user_flows: use this for requests asking for user journeys, flows, steps, or interaction paths\n"
            "- none: use this if the request is planning-related but none of the available tools is a strong fit, and the agent should answer directly\n\n"
            "Reply with only one option name and nothing else.\n"
            "Valid options:\n"
            "generate_mvp_plan\n"
            "extract_core_features\n"
            "define_user_flows\n"
            "none\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        valid_options = {
            "generate_mvp_plan",
            "extract_core_features",
            "define_user_flows",
            "none",
        }

        if selected_tool not in valid_options:
            return "none"

        return selected_tool

    def respond_directly(self, user_message: str) -> str:
        prompt = (
            "You are the Planner Agent of a software assistant system.\n"
            "Answer the user's request directly without using any tool.\n"
            "You specialize in product planning tasks such as MVP scope, feature prioritization, "
            "user flows, roadmap thinking, and product structure.\n"
            "Be clear, practical, and well organized.\n"
            "Do not mention internal tools, routing, or system behavior.\n\n"
            f"User request: {user_message}"
        )

        return self.llm_client.generate(prompt)

    def handle(self, user_message: str) -> AgentResult:
        selected_tool = self.choose_tool(user_message)

        if selected_tool == "none":
            response = self.respond_directly(user_message)

            return AgentResult(
                agent_name=self.name,
                content=response,
                used_tools=[]
            )

        tool_function = self.tools[selected_tool]
        tool_result = tool_function(self.llm_client, user_message)

        return AgentResult(
            agent_name=self.name,
            content=tool_result,
            used_tools=[selected_tool]
        )