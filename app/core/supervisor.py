from collections.abc import Callable, Iterator
from typing import Optional

from app.agents.api_agent import APIAgent
from app.agents.database_agent import DatabaseAgent
from app.agents.general_agent import GeneralAgent
from app.agents.planner_agent import PlannerAgent
from app.core.llm_client import LLMClient
from app.schemas.messages import SupervisorResponse
from app.schemas.distilled_task import DistilledTask
from app.schemas.supervisor_context import SupervisorContext
from app.schemas.review_result import ReviewResult


class SupervisorAgent:
    final_response_language_instruction = (
        "Write the final response in Portuguese.\n"
        "Translate headings and explanatory text to Portuguese when needed.\n"
        "Keep code, SQL, endpoint names, field names, and other technical identifiers unchanged when appropriate.\n"
    )

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.recent_messages: list[str] = []
        self.last_selected_agent: str | None = None
        self.last_intent: str | None = None
        self.last_distilled_task: str | None = None

        self.agents = {
            "planner": PlannerAgent(llm_client),
            "api": APIAgent(llm_client),
            "database": DatabaseAgent(llm_client),
            "general": GeneralAgent(llm_client),
        }

    def build_supervisor_context(self, user_message: str) -> SupervisorContext:
        return SupervisorContext(
            user_message=user_message,
            recent_messages=self.recent_messages[-5:],
            last_selected_agent=self.last_selected_agent,
            last_intent=self.last_intent,
            last_distilled_task=self.last_distilled_task,
        )

    def should_skip_distillation(self, user_message: str) -> bool:
        normalized = user_message.strip()
        word_count = len(normalized.split())
        return word_count <= 8

    def choose_agent(self, context: SupervisorContext) -> str:
        prompt = (
            "You are a supervisor agent.\n"
            "Choose the best agent for the user's request.\n"
            "Available agents:\n"
            "- planner: for MVPs, first versions, product planning, product scope, core features, main features, essential capabilities, and user flows\n"
            "- api: for endpoints, requests, responses, DTOs, payloads, and backend/API design\n"
            "- database: for entities, relationships, table design, SQL schemas, and data modeling\n"
            "- general: for general programming questions, explanations, and requests that do not clearly fit the other agents\n\n"
            "Prefer planner for product planning requests, especially when the user asks about MVPs, first versions, core features, or user flows.\n\n"
            "Use the available conversation context when relevant.\n"
            "The current user request has priority over previous context.\n"
            "Use previous context only when it helps clarify an ambiguous follow-up.\n"
            "If the current request is clear on its own, ignore unrelated previous context.\n\n"
            "If the request is a general knowledge question, a casual factual question, or clearly outside product planning, API design, and database design, choose general.\n"
            "Do not force specialized agents for unrelated topics.\n\n"
            f"Recent messages: {context.recent_messages if context.recent_messages else 'None'}\n"
            f"Last selected agent: {context.last_selected_agent if context.last_selected_agent else 'None'}\n"
            f"Last intent: {context.last_intent if context.last_intent else 'None'}\n"
            f"Last distilled task: {context.last_distilled_task if context.last_distilled_task else 'None'}\n\n"
            "Reply with only one word: planner, api, database, or general.\n\n"
            f"User request: {context.user_message}"
        )

        selected_agent = self.llm_client.generate(prompt).lower().strip()

        if selected_agent not in self.agents:
            return "general"

        return selected_agent

    def build_distilled_task(self, context: SupervisorContext, selected_agent: str) -> DistilledTask:
        if self.should_skip_distillation(context.user_message):
            return DistilledTask(
                original_message=context.user_message,
                distilled_prompt=context.user_message.strip(),
                selected_agent=selected_agent,
                intent=None,
                constraints=[],
            )

        prompt = (
            "You are a supervisor agent preparing a task for a specialized worker agent.\n"
            "Your job is to rewrite the user's request into a shorter, clearer, more focused task for the selected agent.\n"
            "Keep only information that is useful for solving the request.\n"
            "Preserve important constraints, preferences, scope, and key technical details.\n"
            "Remove conversational filler, repetition, and irrelevant background.\n\n"
            "Use the available conversation context when relevant.\n"
            "The current user request has priority over previous context.\n"
            "Use previous context only to resolve ambiguity in short follow-up messages.\n"
            "If the current request is already clear, do not inject unrelated prior topics.\n"
            "Prefer a direct worker instruction style.\n"
            "Preserve the specific project domain from the current and previous context when relevant.\n\n"
            f"Recent messages: {context.recent_messages if context.recent_messages else 'None'}\n"
            f"Last selected agent: {context.last_selected_agent if context.last_selected_agent else 'None'}\n"
            f"Last intent: {context.last_intent if context.last_intent else 'None'}\n"
            f"Last distilled task: {context.last_distilled_task if context.last_distilled_task else 'None'}\n\n"
            "Also infer a short intent label when possible.\n\n"
            "Return your answer in exactly this format:\n"
            "DISTILLED_PROMPT: <short focused task>\n"
            "INTENT: <short intent label or none>\n"
            "CONSTRAINTS: <comma-separated constraints or none>\n\n"
            f"Selected agent: {selected_agent}\n"
            f"User request: {context.user_message}"
        )

        raw_output = self.llm_client.generate(prompt).strip()

        distilled_prompt = context.user_message.strip()
        intent = None
        constraints: list[str] = []

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("DISTILLED_PROMPT:"):
                distilled_prompt = line.removeprefix("DISTILLED_PROMPT:").strip()

            elif line.startswith("INTENT:"):
                raw_intent = line.removeprefix("INTENT:").strip()
                if raw_intent and raw_intent.lower() != "none":
                    intent = raw_intent

            elif line.startswith("CONSTRAINTS:"):
                raw_constraints = line.removeprefix("CONSTRAINTS:").strip()
                if raw_constraints and raw_constraints.lower() != "none":
                    constraints = [
                        item.strip()
                        for item in raw_constraints.split(",")
                        if item.strip()
                    ]

        if not distilled_prompt:
            distilled_prompt = context.user_message.strip()

        return DistilledTask(
            original_message=context.user_message,
            distilled_prompt=distilled_prompt,
            selected_agent=selected_agent,
            intent=intent,
            constraints=constraints,
        )

    def review_agent_result(
        self,
        context: SupervisorContext,
        distilled_task: DistilledTask,
        agent_result,
    ) -> ReviewResult:
        prompt = (
            "You are the supervisor agent of a software assistant system.\n"
            "Your job is to review the worker result before it is returned to the user.\n"
            "Check whether the result is relevant, coherent, and aligned with the user's request.\n"
            "Approve the result if it answers the request well enough.\n"
            "Reject it if it is clearly off-topic, incorrect for the request, too incomplete, or confusing.\n\n"
            "The current user request has priority.\n"
            "Use previous context only when relevant.\n\n"
            "Return your answer in exactly this format:\n"
            "APPROVED: yes or no\n"
            "FEEDBACK: <short correction feedback or none>\n\n"
            f"User request: {context.user_message}\n"
            f"Distilled task: {distilled_task.distilled_prompt}\n"
            f"Intent: {distilled_task.intent if distilled_task.intent else 'None'}\n"
            f"Selected agent: {agent_result.agent_name}\n"
            f"Used tools: {', '.join(agent_result.used_tools) if agent_result.used_tools else 'None'}\n"
            f"Agent result: {agent_result.content}"
        )

        raw_output = self.llm_client.generate(prompt).strip()

        approved = True
        feedback = None

        for line in raw_output.splitlines():
            line = line.strip()

            if line.startswith("APPROVED:"):
                raw_approved = line.removeprefix("APPROVED:").strip().lower()
                approved = raw_approved == "yes"

            elif line.startswith("FEEDBACK:"):
                raw_feedback = line.removeprefix("FEEDBACK:").strip()
                if raw_feedback and raw_feedback.lower() != "none":
                    feedback = raw_feedback

        return ReviewResult(
            approved=approved,
            feedback=feedback,
        )

    def build_retry_task(
        self,
        distilled_task: DistilledTask,
        review_result: ReviewResult,
    ) -> DistilledTask:
        return DistilledTask(
            original_message=distilled_task.original_message,
            distilled_prompt=(
                "Retry the same request with the following correction.\n"
                f"Original task: {distilled_task.distilled_prompt}\n"
                f"Correction feedback: {review_result.feedback}\n"
                "Do not change the task domain.\n"
                "Do not introduce unrelated topics.\n"
                "Only improve the answer for the same request."
            ),
            selected_agent=distilled_task.selected_agent,
            intent=distilled_task.intent,
            constraints=distilled_task.constraints,
        )

    def handle(
        self,
        user_message: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> SupervisorResponse:
        context = self.build_supervisor_context(user_message)

        selected_agent = self.choose_agent(context)
        if progress_callback:
            progress_callback(f"Selected agent: {selected_agent}")

        distilled_task = self.build_distilled_task(context, selected_agent)
        if progress_callback:
            progress_callback(f"Distilled task: {distilled_task.distilled_prompt}")
        if progress_callback and distilled_task.intent:
            progress_callback(f"Intent: {distilled_task.intent}")

        agent = self.agents.get(selected_agent, self.agents["general"])

        max_retries = 1
        attempt = 0

        agent_result = agent.handle(distilled_task, progress_callback=progress_callback)
        review_result = self.review_agent_result(context, distilled_task, agent_result)

        if progress_callback:
            progress_callback(f"Review approved: {review_result.approved}")
        if progress_callback and review_result.feedback:
            progress_callback(f"Review feedback: {review_result.feedback}")

        while not review_result.approved and attempt < max_retries:
            attempt += 1

            if progress_callback:
                progress_callback(f"Retrying agent (attempt {attempt + 1})...")

            corrected_task = self.build_retry_task(distilled_task, review_result)

            agent_result = agent.handle(corrected_task, progress_callback=progress_callback)
            review_result = self.review_agent_result(context, corrected_task, agent_result)

            if progress_callback:
                progress_callback(f"Review approved: {review_result.approved}")
            if progress_callback and review_result.feedback:
                progress_callback(f"Review feedback: {review_result.feedback}")

            distilled_task = corrected_task

        final_prompt = (
            "You are the supervisor agent of a software assistant system.\n"
            "Your job is to produce the final response for the user.\n"
            "Use the agent result as the main source of truth.\n"
            "You may lightly reformulate the text for clarity, readability, and flow.\n"
            "Do not significantly expand the scope.\n"
            "Do not introduce major new ideas.\n"
            "Keep the final answer aligned with the agent result.\n"
            "Preserve the original structure whenever possible.\n\n"
            f"{self.final_response_language_instruction}\n"
            f"User request: {context.user_message}\n"
            f"Distilled task: {distilled_task.distilled_prompt}\n"
            f"Intent: {distilled_task.intent if distilled_task.intent else 'None'}\n"
            f"Constraints: {', '.join(distilled_task.constraints) if distilled_task.constraints else 'None'}\n"
            f"Selected agent: {agent_result.agent_name}\n"
            f"Used tools: {', '.join(agent_result.used_tools) if agent_result.used_tools else 'None'}\n"
            f"Agent result: {agent_result.content}\n\n"
            "Write the final response for the user."
        )

        if progress_callback:
            progress_callback("Generating final response...")

        final_response = self.llm_client.generate(final_prompt)

        self.recent_messages.append(user_message)
        self.last_selected_agent = selected_agent
        self.last_intent = distilled_task.intent
        self.last_distilled_task = distilled_task.distilled_prompt

        return SupervisorResponse(
            selected_agent=agent_result.agent_name,
            final_response=final_response,
            used_tools=agent_result.used_tools,
        )

    def handle_stream(
        self,
        user_message: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[dict]:
        context = self.build_supervisor_context(user_message)

        selected_agent = self.choose_agent(context)
        if progress_callback:
            progress_callback(f"Selected agent: {selected_agent}")

        distilled_task = self.build_distilled_task(context, selected_agent)
        if progress_callback:
            progress_callback(f"Distilled task: {distilled_task.distilled_prompt}")
        if progress_callback and distilled_task.intent:
            progress_callback(f"Intent: {distilled_task.intent}")

        agent = self.agents.get(selected_agent, self.agents["general"])

        max_retries = 1
        attempt = 0

        agent_result = agent.handle(distilled_task, progress_callback=progress_callback)
        review_result = self.review_agent_result(context, distilled_task, agent_result)

        if progress_callback:
            progress_callback(f"Review approved: {review_result.approved}")
        if progress_callback and review_result.feedback:
            progress_callback(f"Review feedback: {review_result.feedback}")

        while not review_result.approved and attempt < max_retries:
            attempt += 1

            if progress_callback:
                progress_callback(f"Retrying agent (attempt {attempt + 1})...")

            corrected_task = self.build_retry_task(distilled_task, review_result)

            agent_result = agent.handle(corrected_task, progress_callback=progress_callback)
            review_result = self.review_agent_result(context, corrected_task, agent_result)

            if progress_callback:
                progress_callback(f"Review approved: {review_result.approved}")
            if progress_callback and review_result.feedback:
                progress_callback(f"Review feedback: {review_result.feedback}")

            distilled_task = corrected_task

        final_prompt = (
            "You are the supervisor agent of a software assistant system.\n"
            "Your job is to produce the final response for the user.\n"
            "Use the agent result as the main source of truth.\n"
            "You may lightly reformulate the text for clarity, readability, and flow.\n"
            "Do not significantly expand the scope.\n"
            "Do not introduce major new ideas.\n"
            "Keep the final answer aligned with the agent result.\n"
            "Preserve the original structure whenever possible.\n\n"
            f"{self.final_response_language_instruction}\n"
            f"User request: {context.user_message}\n"
            f"Distilled task: {distilled_task.distilled_prompt}\n"
            f"Intent: {distilled_task.intent if distilled_task.intent else 'None'}\n"
            f"Constraints: {', '.join(distilled_task.constraints) if distilled_task.constraints else 'None'}\n"
            f"Selected agent: {agent_result.agent_name}\n"
            f"Used tools: {', '.join(agent_result.used_tools) if agent_result.used_tools else 'None'}\n"
            f"Agent result: {agent_result.content}\n\n"
            "Write the final response for the user."
        )

        if progress_callback:
            progress_callback("Generating final response...")

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

        self.recent_messages.append(user_message)
        self.last_selected_agent = selected_agent
        self.last_intent = distilled_task.intent
        self.last_distilled_task = distilled_task.distilled_prompt

        yield {
            "type": "done",
            "selected_agent": agent_result.agent_name,
            "used_tools": agent_result.used_tools,
            "final_response": "".join(final_response_parts),
        }