from pydantic import BaseModel, Field
from typing import List, Optional


class DistilledTask(BaseModel):
    original_message: str
    distilled_prompt: str
    selected_agent: str
    intent: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)