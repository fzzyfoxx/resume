import pytest
from fcgb.types.tavily import TavilySearchSingleResult, TavilySearchResults, TavilyExtractSingleResult, TavilyExtractResults
from fcgb.fake_models import FakeTavily

def test_fake_tavily_search_single_result():
    """
    Test the FakeTavily search single result generation.
    """
    # Create an instance of the FakeTavily
    tavily = FakeTavily()
    
    # Generate a fake search single result
    result = tavily._set_search_result(1)
    
    # Check if the result is of the expected type
    assert list(result.keys()) == list(TavilySearchSingleResult.__annotations__.keys())

def test_fake_tavily_search_results():
    """
    Test the FakeTavily search results generation.
    """
    # Create an instance of the FakeTavily
    tavily = FakeTavily()
    
    # Generate fake search results
    results = tavily.search(query="test", max_results=3)
    
    # Check if the result is of the expected type
    assert list(results.keys()) == list(TavilySearchResults.__annotations__.keys())

def test_fake_tavily_extract_single_result():
    """
    Test the FakeTavily extract single result generation.
    """
    # Create an instance of the FakeTavily
    tavily = FakeTavily()
    
    # Generate a fake extract single result
    result = tavily._set_extract_result("https://fake-url.com", 1)
    
    # Check if the result is of the expected type
    assert list(result.keys()) == list(TavilyExtractSingleResult.__annotations__.keys())

def test_fake_tavily_extract_results():
    """
    Test the FakeTavily extract results generation.
    """
    # Create an instance of the FakeTavily
    tavily = FakeTavily()
    
    # Generate fake extract results
    results = tavily.extract(urls=["https://fake-url.com", "https://fake-url2.com"])
    
    # Check if the result is of the expected type
    assert list(results.keys()) == list(TavilyExtractResults.__annotations__.keys())
