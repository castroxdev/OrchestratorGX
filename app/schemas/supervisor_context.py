from pydantic import BaseModel, Field
from typing import List, Optional


class SupervisorContext(BaseModel):
    # Contexto curto usado pelo supervisor para decidir routing e distilação
    # sem depender de um histórico completo da conversa.
    user_message: str
    recent_messages: List[str] = Field(default_factory=list)
    last_selected_agent: Optional[str] = None
    last_intent: Optional[str] = None
    last_distilled_task: Optional[str] = None
