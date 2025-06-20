from pydantic import BaseModel, Field
from typing import Literal, List

class ToolOutput(BaseModel):
    thread_id: str
    output: str

class JobVarModel(BaseModel):
    job: str
    restrictions: str
    output_format: str
    variables: List[Literal['knowledge_base', 'restrictions']] = Field(..., description="List of additional context variables")

class JobsVarListModel(BaseModel):
    jobs: List[JobVarModel]

class DescriptiveJobsListModel(BaseModel):
    jobs: List[str] = Field(..., description="List of planned job descriptions")

class PlannedTaskRoutingModel(BaseModel):
    decision: Literal['continue', 'report']