from pydantic import BaseModel


class AgentResult(BaseModel):
    agent_name: str
    content: str
    used_tool: str | None = None


class SupervisorResponse(BaseModel):
    selected_agent: str
    final_response: str