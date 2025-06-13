from pydantic import BaseModel, Field
from typing import Literal, List

class ToolOutput(BaseModel):
    thread_id: str
    output: str

class JobModel(BaseModel):
    job: str
    restrictions: str
    output_format: str
    variables: List[Literal['knowledge_base', 'restrictions']] = Field(..., description="List of additional context variables")

class JobsListModel(BaseModel):
    jobs: List[JobModel]

class PlannedTaskRoutingModel(BaseModel):
    decision: Literal['continue', 'report']