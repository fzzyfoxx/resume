from typing import List
from pydantic import BaseModel, Field

class MainSubjectModel(BaseModel):
    main_subject: str

class SubjectDetailsModel(BaseModel):
    content: str
    style: str
    target_audience: str
    layout: str
    restrictions: List[str]

class PersonaModel(BaseModel):
    nickname: str = Field(description="Nickname of the persona, can contain some name and an adjective representing the persona's role")
    proffesion_description: str = Field(description="Description of the persona's profession")
    knowledge_level: str = Field(description="Knowledge level of the persona in context of the book main subject")
    motivation: str = Field(description="What persona would like to achieve by reading the book, what are their goals")

class WorkersModel(BaseModel):
    workers: List[PersonaModel]