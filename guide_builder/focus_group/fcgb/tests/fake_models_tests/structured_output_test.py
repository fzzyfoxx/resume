import pytest
from fcgb.types.example_structures import simple_model, model_with_nesting, model_with_double_nesting
from fcgb.fake_models import FakeStructuredOutput

@pytest.mark.parametrize("input_model", 
                         [simple_model, model_with_nesting, model_with_double_nesting], 
                         ids=["simple_model", "model_with_nesting", "model_with_double_nesting"]
                         )
def test_fake_structured_output(input_model):
    """
    Test the FakeStructuredOutput generator for various Pydantic models.
    """
    generator = FakeStructuredOutput(input_model)
    
    # Generate fake data
    fake_data = generator()
    
    # Check if the generated data is of the expected type
    assert isinstance(fake_data, input_model)
    
    # Check if the generated data contains all required fields
    for field in input_model.model_fields:
        assert hasattr(fake_data, field)

