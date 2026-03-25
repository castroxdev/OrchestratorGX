from pydantic import BaseModel

class AgentResult(BaseModel):
    agent_name:str
    content: str

class SupervisorResponse(BaseModel):
    selected_agent: str
    final_response: str
    