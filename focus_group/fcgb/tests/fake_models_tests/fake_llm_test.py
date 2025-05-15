import pytest
from langchain_core.messages import AIMessage
from fcgb.types.example_structures import simple_model, model_with_nesting, model_with_double_nesting
from fcgb.fake_models import FakeLLM

@pytest.fixture
def fake_llm_invoke():
    """
    Fixture to retreive FakeLLM invoke output
    """
    # Create an instance of the FakeLLM
    llm = FakeLLM()
    
    # Define a prompt
    prompt = "What is the capital of France?"
    
    # Invoke the LLM with the prompt
    response = llm.invoke(prompt)
    
    return response

@pytest.fixture
def fake_llm_call():
    """
    Fixture to retreive FakeLLM call output
    """
    # Create an instance of the FakeLLM
    llm = FakeLLM()
    
    # Define a prompt
    prompt = "What is the capital of France?"
    
    # Call the LLM with the prompt
    response = llm(prompt)
    
    return response

@pytest.mark.parametrize("llm_output_fixture", 
                         ["fake_llm_invoke", "fake_llm_call"], 
                         ids=["invoke", "call"]
                         )
def test_fake_llm_casual_output(request, llm_output_fixture):
    """
    Test the FakeLLM generator output for invoke and call methods.
    """
    # Access the fixture output using the request fixture
    llm_output = request.getfixturevalue(llm_output_fixture)
    
    # Check if the output is an instance of AIMessage
    assert isinstance(llm_output, AIMessage)
    
    # Check if the content of the message is a string
    assert isinstance(llm_output.content, str)
    
    # Check if the content is not empty
    assert len(llm_output.content) > 0


@pytest.mark.parametrize("input_model", 
                         [simple_model, model_with_nesting, model_with_double_nesting], 
                         ids=["simple_model", "model_with_nesting", "model_with_double_nesting"]
                         )
def test_fake_llm_structured_output(input_model):
    """
    Test the FakeLLM structured output for various Pydantic models.
    """
    # Create an instance of the FakeLLM
    llm = FakeLLM()
    
    # Define a prompt
    prompt = "Any prompt"

    # Generate fake structured output
    structured_output = llm.with_structured_output(input_model).invoke(prompt)
    
    # Check if the generated data is of the expected type
    assert isinstance(structured_output, input_model)
    
    # Check if the generated data contains all required fields
    for field in input_model.model_fields:
        assert hasattr(structured_output, field)