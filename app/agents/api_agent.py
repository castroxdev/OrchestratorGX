from app.core.llm_client import LLMClient
from app.schemas.messages import AgentResult
from app.tools.api.generate_api_endpoints import generate_api_endpoints
from app.tools.api.suggest_request_response_models import suggest_request_response_models


class APIAgent:
    name = "api"

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

        self.tools = {
            "generate_api_endpoints": generate_api_endpoints,
            "suggest_request_response_models": suggest_request_response_models,
        }

    def choose_tool(self, user_message: str) -> str:
        prompt = (
            "You are the API Agent of a software assistant system.\n"
            "Choose the best API tool for the user's request.\n"
            "Available tools:\n"
            "- generate_api_endpoints: for generating REST API endpoints\n"
            "- suggest_request_response_models: for suggesting simple request and response models\n\n"
            "Reply with only one tool name.\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        if selected_tool not in self.tools:
            return "generate_api_endpoints"

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