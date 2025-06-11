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
    llm_kwargs = {
        'max_parallel_tools': 2,
        'tool_usage_prob': 0.5
    }

#------------------------------------------------------------

# Storage config
# name pattern: {storage_engine}_config
# storage RAG config pattern: {storage_engine}_rag_config
# checkpointer config pattern: {checkpointer_mode}_saver_config
class mongodb_config(BaseConfig):
    prod = {
        'db_uri_key': 'MONGO_URI',
        'db_name': 'fcgb-prod',
    }
    dev = {
        'db_uri_key': 'MONGO_URI',
        'db_name': 'fcgb-dev',
    }
    test = {
        'db_uri_key': 'MONGO_URI',
        'db_name': 'fcgb-test',
    }

class mongodb_rag_config(BaseConfig):
    prod = {
        'collection': 'vec-web-data-prod',
        'search_index_name': 'web-data-prod-index',
        'load_web_docs': True,
        'embdding_size': 768,
        'similarity_func': 'cosine',
        'vector_fields': ['relevant_content', 'description'],
        'filter_fields': ['source', 'user_id', 'thread_id']
    }
    dev = {
        'collection': 'vec-web-data-dev',
        'search_index_name': 'web-data-dev-index',
        'load_web_docs': True,
        'embdding_size': 768,
        'similarity_func': 'cosine',
        'vector_fields': ['relevant_content', 'description'],
        'filter_fields': ['source', 'user_id', 'thread_id']
    }
    test = {
        'collection': 'vec-web-data-test',
        'search_index_name': 'web-data-test-index',
        'load_web_docs': True,
        'embdding_size': 768,
        'similarity_func': 'cosine',
        'vector_fields': ['relevant_content', 'description'],
        'filter_fields': ['source', 'user_id', 'thread_id']
    }

class mongodb_saver_config(mongodb_config):
    pass

class local_saver_config(BaseConfig):
    prod = {}
    dev = {}
    test = {}

#------------------------------------------------------------

# Embedding model config
# name pattern: {embedding_model}_embedding_config

class google_embedding_config(BaseConfig):
    embedding_func = VertexAIEmbeddings
    embedding_kwargs = {'model': 'text-embedding-005'}

def none_embedding_func(*args, **kwargs):
    return None

class none_embedding_config(BaseConfig):
    embedding_func = none_embedding_func
    embedding_kwargs = {}

class fake_embedding_config(BaseConfig):
    embedding_func = FakeEmbeddingModel
    embedding_kwargs = {'embedding_size': 768}

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
                'exclude_domains': ['arxiv.org', 'researchgate.net', 'dl.acm.org']
            },
            'extract': {
            }
        }
    max_queries_num = 5

class dev_researcher_config(BaseConfig):
    web_search_kwargs = {
            'search': {
                'max_results': 3,
                'exclude_domains': ['arxiv.org', 'researchgate.net', 'dl.acm.org']
            },
            'extract': {
            }
        }
    max_queries_num = 3

class test_researcher_config(BaseConfig):
    web_search_kwargs = {
            'search': {
                'max_results': 3,
                'exclude_domains': ['arxiv.org', 'researchgate.net', 'dl.acm.org']
            },
            'extract': {
            }
        }
    max_queries_num = 3

#------------------------------------------------------------





    