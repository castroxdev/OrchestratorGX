from pydantic import BaseModel
from typing import Optional


class ReviewResult(BaseModel):
    # Guarda a decisão do review do supervisor e o eventual feedback para retry.
    approved: bool
    feedback: Optional[str] = None
