from pydantic import BaseModel

class ToolOutput(BaseModel):
    thread_id: str
    output: str

class JobModel(BaseModel):
    job: str
    motivation: str
    restrictions: str
    output_format: str

class JobsListModel(BaseModel):
    jobs: list[JobModel]