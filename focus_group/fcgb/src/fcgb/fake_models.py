from typing import Any, List, Dict, Tuple, TypedDict, get_type_hints
from pydantic import BaseModel
import random
from langchain_core.messages import AIMessage
from numpy.random import uniform
import string
from fcgb.types.tavily import TavilySearchSingleResult, TavilySearchResults, TavilyExtractSingleResult, TavilyExtractResults

def random_string(length: int) -> str:
    """
    Generate a random string of fixed length.
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class FakeStructuredOutput:
    def __init__(self, model: BaseModel, array_length_range: Tuple[int, int] = (1, 4)):
        """
        Initialize the FakeStructuredOutput generator.

        Args:
            model (BaseModel): The Pydantic BaseModel class to generate fake data for.
            array_length_range (Tuple[int, int]): Range for the length of generated lists.
        """
        self.model = model
        self.array_length_range = array_length_range

    def __call__(self) -> Any:
        """
        Generate fake structured output based on the model schema.

        Returns:
            Any: A dictionary or value matching the model's structure.
        """
        return self._generate_field(self.model)

    def _generate_field(self, field_type: Any) -> Any:
        """
        Generate a fake value for a given field type.

        Args:
            field_type (Any): The type of the field.

        Returns:
            Any: A fake value matching the field type.
        """
        if hasattr(field_type, "__origin__"):  # Handle generic types like List, Dict, etc.
            if field_type.__origin__ is list:
                return self._generate_list(field_type.__args__[0])
            elif field_type.__origin__ is dict:
                return self._generate_dict(field_type.__args__[0], field_type.__args__[1])
            elif field_type.__origin__ is tuple:
                return self._generate_tuple(field_type.__args__)
        elif isinstance(field_type, type):  # Ensure field_type is a class
            # Check if field_type is a TypedDict
            if hasattr(field_type, "__annotations__") and issubclass(field_type, dict):
                return self._generate_typed_dict(field_type)
            elif issubclass(field_type, BaseModel):  # Handle nested BaseModel
                return self._generate_model(field_type)
        if (field_type is str) | (field_type is Any):
            return self._get_fake_string()
        elif field_type is int:
            return self._get_fake_int()
        elif field_type is float:
            return self._get_fake_float()
        elif field_type is bool:
            return self._get_fake_bool()
        elif field_type is None:
            return None
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
        
    def _generate_typed_dict(self, typed_dict: TypedDict) -> Dict[str, Any]: # type: ignore
        """
        Generate fake data for a TypedDict.

        Args:
            typed_dict (TypedDict): The TypedDict class.

        Returns:
            Dict[str, Any]: A dictionary with fake data matching the TypedDict structure.
        """
        type_hints = get_type_hints(typed_dict)
        return {key: self._generate_field(value) for key, value in type_hints.items()}

    def _generate_model(self, model: BaseModel) -> BaseModel:
        """
        Generate fake data for a nested BaseModel.

        Args:
            model (BaseModel): The nested BaseModel class.

        Returns:
            BaseModel: An instance of the model with fake data.
        """
        data = {field_name: self._generate_field(field_type)
                for field_name, field_type in model.__annotations__.items()}
        return model(**data)

    def _generate_list(self, item_type: Any) -> List[Any]:
        """
        Generate a fake list of items.

        Args:
            item_type (Any): The type of items in the list.

        Returns:
            List[Any]: A list of fake items.
        """
        length = random.randint(*self.array_length_range)
        return [self._generate_field(item_type) for _ in range(length)]

    def _generate_dict(self, key_type: Any, value_type: Any) -> Dict[Any, Any]:
        """
        Generate a fake dictionary.

        Args:
            key_type (Any): The type of keys in the dictionary.
            value_type (Any): The type of values in the dictionary.

        Returns:
            Dict[Any, Any]: A dictionary with fake keys and values.
        """
        length = random.randint(*self.array_length_range)
        return {self._generate_field(key_type): self._generate_field(value_type) for _ in range(length)}

    def _generate_tuple(self, item_types: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """
        Generate a fake tuple.

        Args:
            item_types (Tuple[Any, ...]): The types of items in the tuple.

        Returns:
            Tuple[Any, ...]: A tuple of fake items.
        """
        return tuple(self._generate_field(item_type) for item_type in item_types)

    def _get_fake_string(self) -> str:
        """Generate a fake string."""
        return f"Fake string {random_string(5)}"

    def _get_fake_int(self) -> int:
        """Generate a fake integer."""
        return random.randint(0, 100)

    def _get_fake_float(self) -> float:
        """Generate a fake float."""
        return random.uniform(0.0, 100.0)

    def _get_fake_bool(self) -> bool:
        """Generate a fake boolean."""
        return random.choice([True, False])
    

class FakeLLM:
    """
        A fake LLM that returns a simple string response.
        This class is used for testing purposes and simulates the behavior of a language model.
        It increments a counter each time it is called and returns a string indicating the response number.
        The class also provides an `invoke` method that behaves the same as the `__call__` method.    

        Args:
        - args: Additional arguments to be passed to the `__call__` method.
        - kwargs: Additional keyword arguments to be passed to the `__call__` method.
        
        Returns:
        - A string indicating the response number.
    """
    def __init__(self, *args, **kwargs):
        self.counter = 0
        self.structured_output = None

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.structured_output:
            response = self.structured_output
            self.structured_output = None
            return response
        return AIMessage(f"Fake LLM response {self.counter}")
    
    def invoke(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    @staticmethod
    def generate_fake_output(model: BaseModel):
        return FakeStructuredOutput(model)()
    
    def with_structured_output(self, structure):
        self.structured_output = self.generate_fake_output(structure)

        return self
    

class FakeHuman:
    """
        A fake human input generator that returns a simple string response.
        This class is used for testing purposes and simulates the behavior of a human input generator.
        It increments a counter each time it is called and returns a string indicating the response number.
        The class also provides an `__call__` method that can return a specific button output if provided.

        Args:
        - button_output: A string that represents the output of a button.
        - button_moment: An integer that represents the moment when the button output should be returned.

        Returns:
        - A dictionary containing the type of message and its value.
    """
    def __init__(self, button_output=None, button_moment=None):
        self.counter = 0
        self.button_output = button_output
        self.button_moment = button_moment

    def __call__(self, button=None):
        self.counter += 1
        if button:
            return {'type': 'button', 'value': button}
        elif (self.button_output is not None) & (self.button_moment==self.counter):
            return {'type': 'button', 'value': self.button_output}
        return {'type': 'message', 'value': f"Fake Human input {self.counter}"}
    

class FakeEmbeddingModel:
    """
        A fake embedding model that generates random embeddings.
        This class is used for testing purposes and simulates the behavior of an embedding model.
    """
    def __init__(self, embedding_size=768):
        self.embedding_size = embedding_size

    def _gen_embs(self):
        return uniform(-1, 1, self.embedding_size).tolist()

    def embed_documents(self, texts, *args, **kwargs):
        return [self._gen_embs() for _ in texts]
    
    def embed_query(self, text, *args, **kwargs):
        return self._gen_embs()


class FakeTavily:
    def __init__(self):
        pass

    def _set_search_result(self, i: int) -> TavilySearchSingleResult:
        return {
            "title": f"Fake title {i} {random_string(5)}",
            "url": f"https://fake-url-{i}-{random_string(5)}.com",
            "content": f"Fake content {i}-{random_string(5)}",
            "score": uniform(0, 1),
            "raw_content": None
        }
    
    def _set_extract_result(self, url: str, i: int) -> TavilyExtractSingleResult:
        return {
            "url": url,
            "raw_content": f"Fake raw content {i} {random_string(5)}",
            "images": []
        }

    def search(self, query: str, max_results: int, exclude_domains: List[str]=None, *args, **kwargs) -> TavilySearchResults:
        results = [self._set_search_result(i) for i in range(max_results)]

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": None,
            "results": results,
            "response_time": uniform(0, 1)
        }
    
    def extract(self, urls: List[str], *args, **kwargs) -> TavilyExtractResults:
        results = [self._set_extract_result(url, i) for i, url in enumerate(urls)]

        return {
            "results": results,
            "failed_results": [],
            "response_time": uniform(0, 1)
        }