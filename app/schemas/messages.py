from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    # This is the payload that an agent returns to the supervisor after either
    # answering directly or chaining one or more tools.
    agent_name: str
    content: str
    used_tools: list[str] = Field(default_factory=list)


class SupervisorResponse(BaseModel):
    # This is the final payload returned by the supervisor to the CLI or web UI.
    # It keeps the chosen agent and used tools visible without exposing internals.
    selected_agent: str
    final_response: str
    used_tools: list[str] = Field(default_factory=list)
