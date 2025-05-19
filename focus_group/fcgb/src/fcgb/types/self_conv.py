from typing import List
from pydantic import BaseModel

class SelfConvModel(BaseModel):
    answer: str