from pydantic import BaseModel, Field
from typing import List, Optional


class SupervisorContext(BaseModel):
    user_message: str
    recent_messages: List[str] = Field(default_factory=list)
    last_selected_agent: Optional[str] = None
    last_intent: Optional[str] = None
    last_distilled_task: Optional[str] = None