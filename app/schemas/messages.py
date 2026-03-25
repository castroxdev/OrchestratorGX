from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    agent_name: str
    content: str
    used_tools: list[str] = Field(default_factory=list)


class SupervisorResponse(BaseModel):
    selected_agent: str
    final_response: str
    used_tools: list[str] = Field(default_factory=list)