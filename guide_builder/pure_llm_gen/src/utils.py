import json
import random
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import JsonOutputParser
from importlib import import_module

def save_to_json(input_string, filename):
    with open(filename, 'w') as json_file:
        json.dump(input_string, json_file, ensure_ascii=False, indent=4)

def load_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)
    
def multiline_print(text):
    for line in text.split("\n"):
        print(line)

def assign_ids(sections, key="id", format=int, starting_num=0):
    return [{key: format(i+starting_num), **elem} for i, elem in enumerate(sections)]

def format_with_exception(x, target_format, fill):
    try:
        return target_format(x)
    except:
        return fill

def assign_formats(input_dict, specs):
    if type(input_dict) is not dict:
        return []
    existing_keys = input_dict.keys()
    output_dict = {}
    for spec in specs:
        key = spec['key']
        if key in existing_keys:
            output_dict[key] = format_with_exception(x=input_dict[key], target_format=spec['format'], fill=spec['fill'])
        else:
            output_dict[key] = spec['fill']
    return [output_dict]

def set_dicts_to_proper_format(input_dicts, specs):
    if type(input_dicts) is not list:
        input_dicts = [input_dicts]

    return sum([assign_formats(input_dict, specs) for input_dict in input_dicts], [])

def create_path_if_not_exists(path):
    try:
        os.makedirs(path)
    except:
        pass

def get_siblings_for_elem(elem, sections, sep=', ', key='current_sections'):
    parent_id = elem['parent_id']
    sibling_sections = sep.join([x['title'] + '-' + x['subtitle'] for x in sections if x['parent_id'] == parent_id])
    output = elem.copy()
    output[key] = sibling_sections
    return output

def get_siblings(sections, sep='\n', key='current_sections'):
    return [get_siblings_for_elem(elem, sections.copy(), sep, key) for elem in sections]

def count_files_in_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        print(f"Number of files in the folder: {len(files)}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

class RandomNameGenerator:
    def __init__(self):
        self.attributes = [
                "Agile", "Alert", "Ambitious", "Attentive", "Bold", "Brave", "Calm", "Cautious", "Clever", "Confident",
                "Curious", "Daring", "Diligent", "Energetic", "Fearless", "Fierce", "Friendly", "Gentle", "Graceful", "Hardy",
                "Honest", "Intelligent", "Loyal", "Majestic", "Mighty", "Noble", "Observant", "Patient", "Playful", "Powerful",
                "Quick", "Quiet", "Resourceful", "Robust", "Savvy", "Sly", "Smart", "Spirited", "Stealthy", "Strong",
                "Swift", "Tenacious", "Tough", "Valiant", "Vigilant", "Wise", "Witty", "Zealous", "Adaptable", "Adventurous",
                "Affectionate", "Agile", "Alert", "Amiable", "Assertive", "Attentive", "Brave", "Bright", "Calm", "Caring",
                "Cautious", "Charming", "Cheerful", "Clever", "Confident", "Courageous", "Curious", "Daring", "Diligent", "Eager",
                "Energetic", "Enthusiastic", "Fearless", "Fierce", "Friendly", "Gentle", "Graceful", "Hardworking", "Honest", "Humble",
                "Independent", "Ingenious", "Intelligent", "Joyful", "Kind", "Lively", "Loyal", "Majestic", "Mighty", "Noble",
                "Observant", "Patient", "Playful", "Powerful", "Quick", "Quiet", "Resourceful", "Robust", "Savvy", "Sly"
            ]
        self.animal_names = [
                "Aardvark", "Albatross", "Alligator", "Alpaca", "Ant", "Anteater", "Antelope", "Ape", "Armadillo", "Donkey",
                "Baboon", "Badger", "Barracuda", "Bat", "Bear", "Beaver", "Bee", "Bison", "Boar", "Buffalo",
                "Butterfly", "Camel", "Capybara", "Caribou", "Cassowary", "Cat", "Caterpillar", "Cattle", "Chamois", "Cheetah",
                "Chicken", "Chimpanzee", "Chinchilla", "Chough", "Clam", "Cobra", "Cockroach", "Cod", "Cormorant", "Coyote",
                "Crab", "Crane", "Crocodile", "Crow", "Curlew", "Deer", "Dinosaur", "Dog", "Dogfish", "Dolphin",
                "Dotterel", "Dove", "Dragonfly", "Duck", "Dugong", "Dunlin", "Eagle", "Echidna", "Eel", "Eland",
                "Elephant", "Elk", "Emu", "Falcon", "Ferret", "Finch", "Fish", "Flamingo", "Fly", "Fox",
                "Frog", "Gaur", "Gazelle", "Gerbil", "Giraffe", "Gnat", "Gnu", "Goat", "Goldfinch", "Goldfish",
                "Goose", "Gorilla", "Goshawk", "Grasshopper", "Grouse", "Guanaco", "Gull", "Hamster", "Hare", "Hawk",
                "Hedgehog", "Heron", "Herring", "Hippopotamus", "Hornet", "Horse", "Human", "Hummingbird", "Hyena", "Ibex"
            ]

    def __call__(self):
        attribute = random.choice(self.attributes)
        animal_name = random.choice(self.animal_names)
        random_integer = random.randint(1000, 9999)
        return f"{attribute}-{animal_name}-{random_integer}"
    

class LLMChatModule:
    def __init__(self, model_def):
        """
        model_def is a dictionary that defines the model configuration.
        It includes the library and module to be used, as well as the arguments for initializing the model.
        Example format:
        model_def = {
            "model_lib": {
            "lib": "transformers",
            "module": "GPT2LMHeadModel"
            },
            "model_args": {
            "pretrained_model_name_or_path": "gpt2"
            }
        }
        """

        self.model_def = model_def
        self.llm = getattr(import_module(model_def["model_lib"]["lib"]), model_def["model_lib"]["module"])(**model_def["model_args"])
            
    def _set_json_fix_chain(self, json_fix_config):

        prompt_spec = self._load_prompt(json_fix_config["prompt_path"])

        jf_template = prompt_spec["template"]
        system_message = prompt_spec["system"]

        chat_prompt = ChatPromptTemplate([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": jf_template.format(task=prompt_spec["task"], context="{context}")}
                ])
        
        self.json_fix_chain = chat_prompt | self.llm | JsonOutputParser()
    
    def _parse_json(self, llm_output) -> Runnable:
        try:
            return JsonOutputParser().invoke(llm_output)
        except:
            return self.json_fix_chain.invoke(llm_output)      
    
    def _get_format_validator(self, specs):
        return RunnableLambda(lambda x: set_dicts_to_proper_format(x, specs=specs))

    def _load_prompt(self, prompt_path):
        with open(prompt_path, 'r') as f:
            prompt = json.load(f)
        return prompt