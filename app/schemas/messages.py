from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    # Contrato intermédio devolvido por um agente ao supervisor.
    agent_name: str
    content: str
    used_tools: list[str] = Field(default_factory=list)


class SupervisorResponse(BaseModel):
    # Payload final exposto pelas entradas externas, como CLI e web.
    selected_agent: str
    final_response: str
    used_tools: list[str] = Field(default_factory=list)
