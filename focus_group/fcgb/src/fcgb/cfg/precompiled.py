import os
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import MemorySaver
import fcgb.cfg.vars as vars
from fcgb.rag.mongodb import MongodbRAG
from fcgb.rag.researcher import RAGResearcher
from typing import Literal

# Functions for base models

def get_embedding_model(embedding_model: Literal['google', 'none', 'fake']):
    """
    Args:
        embedding_model: The type of embedding model to use (google, none, fake).

    Returns:
        An instance of the embedding model.
    """
    embedding_config = getattr(vars, f"{embedding_model}_embedding_config")
    return embedding_config.embedding_func(**embedding_config.embedding_kwargs)

def get_search_engine(search_engine: Literal['tavily', 'fake']):
    """
    Args:
        search_engine: The type of search engine to use (tavily, fake).

    Returns:
        An instance of the search engine.
    """
    search_config = getattr(vars, f"{search_engine}_search_config")
    return search_config.search_func(**search_config.search_kwargs)

def get_llm(llm_model = Literal['google', 'fake']):
    """
    Args:
        llm_model: The type of LLM to use (google, fake).

    Returns:
        An instance of the LLM.
    """
    llm_config = getattr(vars, f"{llm_model}_llm_config")
    return llm_config.llm_func(**llm_config.llm_kwargs)

#------------------------------------------------------------

# Database client functions
# name pattern: _get_{db_engine}_client
def _get_mongodb_client(db_uri_key: str, db_name: str):
    """
    Args:
        db_uri_key: The environment variable key for the MongoDB URI.
        db_name: The name of the database to connect to.

    Returns:
        An instance of the MongoDB database client.
    """
    return MongoClient(os.getenv(db_uri_key))[db_name]

def get_db_client(db_engine: Literal['mongodb'], mode: Literal['prod', 'dev', 'test'] = 'dev'):
    """
    Args:
        db_engine: The database engine to use (mongodb).
        mode: The mode of operation (prod, dev, test).

    Returns:
        An instance of the database client.
    """
    db_config = getattr(getattr(vars, f"{db_engine}_config"), mode)
    db_client_func = globals().get(f"_get_{db_engine}_client")
    return db_client_func(**db_config)

#------------------------------------------------------------

# Get RAG system
# name pattern: _get_{db_engine}_rag

def _get_mongodb_rag(db, mode, embedding_model):
    """
    Args:
        db: The MongoDB database client.
        mode: The mode of operation (prod, dev, test).
        embedding_model: The embedding model to use (google, none, fake).

    Returns:
        An instance of the MongoDB RAG.
    """
    collection_name = getattr(vars.mongodb_rag_config, mode).get('collection')
    return MongodbRAG(
        database=db,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

def get_rag(
        mode: Literal['prod', 'dev', 'test'] = 'dev', 
        db_engine: Literal['mongodb'] = 'mongodb',
        embedding_model: Literal['google', 'none', 'fake'] = 'google'
    ):
    """
    Args:
        mode: The mode of operation (prod, dev, test).
        db_engine: The database engine to use (mongodb).
        embedding_model: The embedding model to use (google, none, fake).

    Returns:
        An instance of the RAG system.
    """
    db = get_db_client(db_engine, mode)
    embedding_model = get_embedding_model(embedding_model)
    rag_func = globals().get(f"_get_{db_engine}_rag")
    return rag_func(db, mode, embedding_model)

#------------------------------------------------------------

# Get LangGraph checkpointer from config
# name pattern: _get_{checkpointer_mode}_saver

def _get_mongodb_saver(db_uri_key: str, db_name: str):
    """
    Args:
        db_uri_key: The environment variable key for the MongoDB URI.
        db_name: The name of the database to connect to.

    Returns:
        An instance of the MongoDB saver.
    """
    db_client = MongoClient(os.getenv(db_uri_key))
    return MongoDBSaver(db_client, db_name=db_name)

def _get_local_saver():
    """
    Returns:
        An instance of the local saver.
    """
    return MemorySaver()

def get_checkpointer(checkpointer_mode: Literal['mongodb', 'local'], mode: Literal['prod', 'dev', 'test'] = 'dev'):
    """
    Args:
        checkpointer_mode: The mode of operation (mongodb, local).
        mode: The mode of operation (prod, dev, test).

    Returns:
        An instance of the checkpointer.
    """
    checkpointer_config = getattr(getattr(vars, f"{checkpointer_mode}_saver_config"), mode)
    saver_func = globals().get(f"_get_{checkpointer_mode}_saver")
    return saver_func(**checkpointer_config)


#------------------------------------------------------------

# Get compiled RAG researcher

def get_researcher(
        mode: Literal['prod', 'dev', 'test'] = 'dev',
        db_engine: Literal['mongodb'] = 'mongodb',
        chackpointer_mode: Literal['mongodb', 'local'] = 'mongodb',
        llm_model: Literal['google', 'fake'] = 'google',
        embedding_model: Literal['google', 'fake'] = 'google',
        search_engine: Literal['tavily', 'fake'] = 'tavily'
    ):
    """
    A function to get a compiled RAG researcher.

    Args:
        mode: The mode of operation (prod, dev, test).
        db_engine: The database engine to use (mongodb).
        chackpointer_mode: The checkpointer mode to use (mongodb, local).
        llm_model: The LLM model to use (google, fake).
        embedding_model: The embedding model to use (google, fake).
        search_engine: The search engine to use (tavily, fake).
    Returns:
        An instance of the compiled RAG researcher.
    """
    
    llm = get_llm(llm_model)
    rag = get_rag(mode, db_engine, embedding_model)
    rag_kwargs = getattr(getattr(vars, f"{db_engine}_rag_config"), mode)
    memory = get_checkpointer(chackpointer_mode, mode)
    search_engine = get_search_engine(search_engine)
    other_kwargs = getattr(vars, f"{mode}_researcher_config").params()

    return RAGResearcher(
        llm=llm,
        rag_module=rag,
        rag_kwargs=rag_kwargs,
        memory=memory,
        web_search_client=search_engine,
        **other_kwargs
    )







