from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from operator import add
from fcgb.types.utils import append_or_clear, extend_or_clear

class QueryListModel(BaseModel):
    queries: List[str]

class WebOutputModel(BaseModel):
    is_relevant: bool = Field(description="Does web page content contains relevant information?")
    relevant_content: str = Field(description="Relevant part for a given context.")
    references: str = Field(description="References to the relevant content.")
    description: str = Field(description="Descriptive summarization of the relevant content.")

class WebDocumentModel(BaseModel):
    is_relevant: bool
    relevant_content: str
    references: str
    description: str
    url: str
    query: str
    thread_id: str
    user_id: str

class RAGGeneralState(BaseModel):
    main_question: str
    current_question: str
    template_inputs: Dict[str, str]
    documents: Annotated[List[WebDocumentModel], append_or_clear]
    retreived_content: Annotated[List[str], add]

class QueriesState(BaseModel):
    current_question: str
    template_inputs: Dict[str, str]
    queries: List[str]

class WebSearchState(BaseModel):
    current_question: str
    query: str
    urls_response: List[Dict[str, Any]] = []

class WebSearchOutputHandlerState(BaseModel):
    current_question: str
    query: str
    url: str
    url_content: str

class WebResponseRoutingModel(BaseModel):
    responses: Annotated[List[WebSearchOutputHandlerState], extend_or_clear]