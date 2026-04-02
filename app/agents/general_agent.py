from collections.abc import Callable
from typing import Optional

from app.core.llm_client import LLMClient
from app.schemas.distilled_task import DistilledTask
from app.schemas.messages import AgentResult


class GeneralAgent:
    # Fallback para pedidos que não encaixam claramente nos agentes
    # especializados disponíveis.
    name = "general"
    response_language_instruction = (
        "Respond in Portuguese.\n"
        "Write all natural-language explanations in Portuguese.\n"
        "Keep code, commands, and technical identifiers unchanged when appropriate.\n"
    )

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def handle(
        self,
        task: DistilledTask,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> AgentResult:
        # Este agente não usa tools: responde diretamente para garantir
        # cobertura do sistema fora dos domínios especializados.
        prompt = (
            "You are the General Agent of a software assistant system.\n"
            "Handle general programming questions, broad explanations, and requests "
            "that do not clearly belong to planning, API design, or database modeling.\n"
            "Reply clearly and directly.\n\n"
            f"{self.response_language_instruction}\n"
            f"User request: {task.distilled_prompt}"
        )

        response = self.llm_client.generate(prompt)

        return AgentResult(
            agent_name=self.name,
            content=response,
            used_tools=[]
        )
