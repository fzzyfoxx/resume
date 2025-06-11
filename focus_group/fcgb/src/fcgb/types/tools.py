from pydantic import BaseModel

class ToolOutput(BaseModel):
    thread_id: str
    output: str