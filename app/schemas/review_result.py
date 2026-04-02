from pydantic import BaseModel
from typing import Optional


class ReviewResult(BaseModel):
    approved: bool
    feedback: Optional[str] = None