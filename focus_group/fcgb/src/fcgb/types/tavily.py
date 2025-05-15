from typing import TypedDict, List, Any

class TavilySearchSingleResult(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float
    raw_content: str | None

class TavilySearchResults(TypedDict, total=False):
    query: str
    follow_up_questions: str | None
    answer: str | None
    images: List[str] | None
    results: List[TavilySearchSingleResult]
    response_time: float

class TavilyExtractSingleResult(TypedDict, total=False):
    url: str
    raw_content: str
    images: List[Any]

class TavilyExtractResults(TypedDict, total=False):
    results: List[TavilyExtractSingleResult]
    failed_results: List[Any]
    response_time: float