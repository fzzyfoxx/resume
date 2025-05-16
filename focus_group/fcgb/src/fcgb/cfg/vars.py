from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from fcgb.fake_models import FakeEmbeddingModel, FakeTavily, FakeLLM
from fcgb.cfg.utils import BaseConfig, extract_named_class_variables
    

# LLms config
# name pattern: {llm_engine}_llm_config
class google_llm_config(BaseConfig):
    llm_func = ChatGoogleGenerativeAI
    llm_kwargs = {
        'model': 'gemini-2.0-flash',
        'temperature': 0.5,
        'max_output_tokens': 2048,
        'max_tokens': 4096
    }

class fake_llm_config(BaseConfig):
    llm_func = FakeLLM
    llm_kwargs = {}

#------------------------------------------------------------

# Storage config
# name pattern: {storage_engine}_config
# storage RAG config pattern: {storage_engine}_rag_config
# checkpointer config pattern: {checkpointer_mode}_saver_config
class mongodb_config(BaseConfig):
    db_name = 'FocusGroup'
    db_uri_key = 'MONGO_URI'

class mongodb_rag_config(BaseConfig):
    prod = {
        'collection': 'vec-web-data',
        'search_index_name': 'web-data-index',
        'load_web_docs': True
    }
    dev = {
        'collection': 'dev-vec-web-data',
        'search_index_name': 'dev-web-data-index',
        'load_web_docs': True
    }
    test = {
        'collection': 'test-vec-web-data',
        'search_index_name': 'test-web-data-index',
        'load_web_docs': True
    }

class mongodb_saver_config(mongodb_config):
    pass

class local_saver_config(BaseConfig):
    pass

#------------------------------------------------------------

# Embedding model config
# name pattern: {embedding_model}_embedding_config

class google_embedding_config(BaseConfig):
    embedding_func = VertexAIEmbeddings
    embedding_kwargs = {'model': 'text-embedding-005'}

class none_embedding_config(BaseConfig):
    embedding_func = None
    embedding_kwargs = {}

class fake_embedding_config(BaseConfig):
    embedding_func = FakeEmbeddingModel
    embedding_kwargs = {}

#------------------------------------------------------------

# Search engine config
# name pattern: {search_engine}_search_config

class tavily_search_config(BaseConfig):
    search_func = TavilyClient
    search_kwargs = {}

class fake_search_config(BaseConfig):
    search_func = FakeTavily
    search_kwargs = {}

#------------------------------------------------------------

# Mode based researcher config
# name pattern: {mode}_researcher_config
class prod_researcher_config(BaseConfig):
    web_search_kwargs = {
            'search': {
                'max_results': 5,
                'exclude_domains': ['arxiv.org', 'researchgate.net', 'dl.acm.org', 'researchgate.net']
            },
            'extract': {
            }
        }
    max_queries_num = 5

class dev_researcher_config(BaseConfig):
    web_search_kwargs = {
            'search': {
                'max_results': 3,
                'exclude_domains': []
            },
            'extract': {
            }
        }
    max_queries_num = 3

class test_researcher_config(BaseConfig):
    web_search_kwargs = {
            'search': {
                'max_results': 3,
                'exclude_domains': []
            },
            'extract': {
            }
        }
    max_queries_num = 3

#------------------------------------------------------------





    