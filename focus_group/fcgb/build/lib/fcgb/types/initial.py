from typing import List
from pydantic import BaseModel

class MainSubjectModel(BaseModel):
    main_subject: str

class SubjectDetailsModel(BaseModel):
    content: str
    style: str
    target_audience: str
    layout: str
    restrictions: List[str]

class PersonaModel(BaseModel):
    nickname: str
    proffesion_description: str
    knowledge_level: str
    motivation: str

class WorkersModel(BaseModel):
    workers: List[PersonaModel]