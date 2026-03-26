from collections.abc import Iterator

from app.agents.api_agent import APIAgent
from app.agents.database_agent import DatabaseAgent
from app.agents.general_agent import GeneralAgent
from app.agents.planner_agent import PlannerAgent
from app.core.llm_client import LLMClient
from app.schemas.messages import SupervisorResponse


class SupervisorAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client: LLMClient = llm_client

        self.agents = {
            "planner": PlannerAgent(llm_client),
            "api": APIAgent(llm_client),
            "database": DatabaseAgent(llm_client),
            "general": GeneralAgent(llm_client),
        }

    def choose_agent(self, user_message: str) -> str:
        prompt = (
            "You are a supervisor agent.\n"
            "Choose the best agent for the user's request.\n"
            "Available agents:\n"
            "- planner: for MVPs, first versions, product planning, product scope, core features, main features, essential capabilities, and user flows\n"
            "- api: for endpoints, requests, responses, DTOs, payloads, and backend/API design\n"
            "- database: for entities, relationships, table design, SQL schemas, and data modeling\n"
            "- general: for general programming questions, explanations, and requests that do not clearly fit the other agents\n\n"
            "Prefer planner for product planning requests, especially when the user asks about MVPs, first versions, core features, or user flows.\n"
            "Reply with only one word: planner, api, database, or general.\n\n"
            f"User request: {user_message}"
        )

        selected_agent = self.llm_client.generate(prompt).lower().strip()

        if selected_agent not in self.agents:
            return "general"

        return selected_agent

    def handle(self, user_message: str) -> SupervisorResponse:
        selected_agent = self.choose_agent(user_message)
        agent = self.agents.get(selected_agent, self.agents["general"])
        agent_result = agent.handle(user_message)

        final_prompt = (
            "You are the supervisor agent of a software assistant system.\n"
            "Your job is to produce the final response for the user.\n"
            "Use the agent result as the main source of truth.\n"
            "You may lightly reformulate the text for clarity, readability, and flow.\n"
            "Do not significantly expand the scope.\n"
            "Do not introduce major new ideas.\n"
            "Keep the final answer aligned with the agent result.\n"
            "Preserve the original structure whenever possible.\n\n"
            f"User request: {user_message}\n"
            f"Selected agent: {agent_result.agent_name}\n"
            f"Used tools: {', '.join(agent_result.used_tools) if agent_result.used_tools else 'None'}\n"
            f"Agent result: {agent_result.content}\n\n"
            "Write the final response for the user."
        )

        final_response = self.llm_client.generate(final_prompt)

        return SupervisorResponse(
            selected_agent=agent_result.agent_name,
            final_response=final_response,
            used_tools=agent_result.used_tools,
        )

    def handle_stream(self, user_message: str) -> Iterator[dict]:
        selected_agent = self.choose_agent(user_message)
        agent = self.agents.get(selected_agent, self.agents["general"])
        agent_result = agent.handle(user_message)

        final_prompt = (
            "You are the supervisor agent of a software assistant system.\n"
            "Your job is to produce the final response for the user.\n"
            "Use the agent result as the main source of truth.\n"
            "You may lightly reformulate the text for clarity, readability, and flow.\n"
            "Do not significantly expand the scope.\n"
            "Do not introduce major new ideas.\n"
            "Keep the final answer aligned with the agent result.\n"
            "Preserve the original structure whenever possible.\n\n"
            f"User request: {user_message}\n"
            f"Selected agent: {agent_result.agent_name}\n"
            f"Used tools: {', '.join(agent_result.used_tools) if agent_result.used_tools else 'None'}\n"
            f"Agent result: {agent_result.content}\n\n"
            "Write the final response for the user."
        )

        yield {
            "type": "metadata",
            "selected_agent": agent_result.agent_name,
            "used_tools": agent_result.used_tools,
        }

        final_response_parts: list[str] = []

        for chunk in self.llm_client.stream_generate(final_prompt):
            final_response_parts.append(chunk)
            yield {
                "type": "chunk",
                "content": chunk,
            }

        yield {
            "type": "done",
            "selected_agent": agent_result.agent_name,
            "used_tools": agent_result.used_tools,
            "final_response": "".join(final_response_parts),
        }
