from collections.abc import Callable
from typing import Optional

from app.core.llm_client import LLMClient
from app.schemas.distilled_task import DistilledTask
from app.schemas.messages import AgentResult
from app.tools.api.generate_api_endpoints import generate_api_endpoints
from app.tools.api.suggest_request_response_models import suggest_request_response_models


class APIAgent:
    name = "api"
    response_language_instruction = (
        "Respond in Portuguese.\n"
        "Write all natural-language explanations, headings, and bullet points in Portuguese.\n"
        "Keep endpoint names, field names, code, and other technical identifiers unchanged when appropriate.\n"
    )

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.tools = {
            "generate_api_endpoints": generate_api_endpoints,
            "suggest_request_response_models": suggest_request_response_models,
        }

    def choose_tools(self, task: DistilledTask) -> list[str]:
        prompt = (
            "You are the API Agent of a software assistant system.\n"
            "Your task is to choose the best API options for the user's request.\n"
            "You may choose one tool, multiple tools, or none.\n"
            "Choose multiple tools only when the request clearly includes multiple distinct API tasks.\n"
            "If the user asks only for API endpoints, choose only generate_api_endpoints.\n"
            "If the user asks only for request/response models, DTOs, payloads, or contracts, choose only suggest_request_response_models.\n"
            "Only choose multiple tools when the user clearly asks for both endpoints and models.\n"
            "For example:\n"
            "- endpoints only -> generate_api_endpoints\n"
            "- request/response models only -> suggest_request_response_models\n"
            "- endpoints + request/response models -> generate_api_endpoints, suggest_request_response_models\n\n"
            "Available options:\n"
            "- generate_api_endpoints: use this for requests asking for API endpoints, routes, resources, controllers, or backend endpoint design\n"
            "- suggest_request_response_models: use this for requests asking for request bodies, response models, DTOs, payloads, contracts, or API data structures\n"
            "- none: use this if the request is API-related but none of the available tools is a strong fit, and the agent should answer directly\n\n"
            "Reply with:\n"
            "- one or more option names separated by commas\n"
            "- or none\n\n"
            "Reply with only valid option names and nothing else.\n\n"
            "Valid options:\n"
            "generate_api_endpoints\n"
            "suggest_request_response_models\n"
            "none\n\n"
            f"User request: {task.distilled_prompt}"
        )

        raw_selection = self.llm_client.generate(prompt).strip().lower()

        if raw_selection == "none":
            return []

        valid_tools = set(self.tools)

        selected_tools: list[str] = []
        for item in raw_selection.split(","):
            tool_name = item.strip()
            if tool_name in valid_tools and tool_name not in selected_tools:
                selected_tools.append(tool_name)

        return selected_tools

    def respond_directly(self, task: DistilledTask) -> str:
        prompt = (
            "You are the API Agent of a software assistant system.\n"
            "Answer the user's request directly without using any tool.\n"
            "You specialize in API design tasks such as endpoints, request and response models, "
            "DTO structure, REST design, backend contracts, and integration decisions.\n"
            "Be clear, practical, and well organized.\n"
            "Do not mention internal tools, routing, or system behavior.\n\n"
            f"{self.response_language_instruction}\n"
            f"User request: {task.distilled_prompt}"
        )

        return self.llm_client.generate(prompt)

    def build_tool_user_request(self, task: DistilledTask) -> str:
        return (
            f"{task.distilled_prompt}\n\n"
            "Mandatory output language: Portuguese.\n"
            "Write all natural-language text, titles, and bullet points in Portuguese.\n"
            "Keep endpoint names, field names, code, and other technical identifiers unchanged when appropriate."
        )

    def combine_tool_results(self, results: list[tuple[str, str]]) -> str:
        sections: list[str] = []

        for tool_name, content in results:
            title = tool_name.replace("_", " ").title()
            sections.append(f"{title}:\n{content}")

        return "\n\n".join(sections)

    def handle(
        self,
        task: DistilledTask,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResult:
        selected_tools = self.choose_tools(task)

        if not selected_tools:
            response = self.respond_directly(task)

            return AgentResult(
                agent_name=self.name,
                content=response,
                used_tools=[],
            )

        tool_results: list[tuple[str, str]] = []
        tool_user_request = self.build_tool_user_request(task)

        if progress_callback:
            progress_callback(f"Selected tools: {', '.join(selected_tools)}")

        for tool_name in selected_tools:
            if progress_callback:
                progress_callback(f"Running tool: {tool_name}")

            tool_function = self.tools[tool_name]
            tool_output = tool_function(self.llm_client, tool_user_request)
            tool_results.append((tool_name, tool_output))

        combined_result = self.combine_tool_results(tool_results)

        return AgentResult(
            agent_name=self.name,
            content=combined_result,
            used_tools=selected_tools,
        )