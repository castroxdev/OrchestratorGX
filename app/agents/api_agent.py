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
            "Your task is to choose the best option for the user's request.\n"
            "Available options:\n"
            "- generate_api_endpoints: use this for requests asking for API endpoints, routes, resources, or backend endpoint design\n"
            "- suggest_request_response_models: use this for requests asking for request bodies, response models, DTOs, payloads, or API data structures\n"
            "- none: use this if the request is API-related but none of the available tools is a strong fit, and the agent should answer directly\n\n"
            "Reply with only one option name and nothing else.\n"
            "Valid options:\n"
            "generate_api_endpoints\n"
            "suggest_request_response_models\n"
            "none\n\n"
            f"User request: {user_message}"
        )

        selected_tool = self.llm_client.generate(prompt).strip().lower()

        valid_options = {
            "generate_api_endpoints",
            "suggest_request_response_models",
            "none",
        }

        if selected_tool not in valid_options:
            return "none"

        return selected_tool

    def respond_directly(self, user_message: str) -> str:
        prompt = (
            "You are the API Agent of a software assistant system.\n"
            "Answer the user's request directly without using any tool.\n"
            "You specialize in API design tasks such as endpoints, request and response models, "
            "DTO structure, REST design, backend contracts, and integration decisions.\n"
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