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
    Get the embedding model based on the specified type.
    :param embedding_model: The type of embedding model to use (google, none, fake).
    :return: An instance of the embedding model.
    """
    embedding_config = getattr(vars, f"{embedding_model}_embedding_config")
    return embedding_config.embedding_func(**embedding_config.embedding_kwargs)

def get_search_engine(search_engine: Literal['tavily', 'fake']):
    """
    Get the search engine based on the specified type.
    :param search_engine: The type of search engine to use (tavily, fake).
    :return: An instance of the search engine.
    """
    search_config = getattr(vars, f"{search_engine}_search_config")
    return search_config.search_func(**search_config.search_kwargs)

def get_llm(llm_model = Literal['google', 'fake']):
    """
    Get the LLM based on the specified type.
    :param llm_model: The type of LLM to use (google, fake).
    :return: An instance of the LLM.
    """
    llm_config = getattr(vars, f"{llm_model}_llm_config")
    return llm_config.llm_func(**llm_config.llm_kwargs)

#------------------------------------------------------------

# Database client functions
# name pattern: _get_{db_engine}_client
def _get_mongodb_client(db_uri_key: str, db_name: str):
    """
    Get the MongoDB client for the specified URI key and database name.
    :param db_uri_key: The environment variable key for the MongoDB URI.
    :param db_name: The name of the database to connect to.
    :return: An instance of the MongoDB database client.
    """
    return MongoClient(os.getenv(db_uri_key))[db_name]

def get_db_client(db_engine: Literal['mongodb']):
    """
    Get the database client for the specified engine.
    :param db_engine: The database engine to use (mongodb).
    :return: An instance of the database client.
    """
    db_config = getattr(vars, f"{db_engine}_config")
    db_client_func = globals().get(f"_get_{db_engine}_client")
    return db_client_func(**db_config.params())

#------------------------------------------------------------

# Get RAG system
# name pattern: _get_{db_engine}_rag

def _get_mongodb_rag(db, mode, embedding_model):
    """
    Get the MongoDB RAG instance for the specified mode and embedding model.
    :param db: The MongoDB database client.
    :param mode: The mode of operation (prod, dev, test).
    :param embedding_model: The embedding model to use (google, none, fake).
    :return: An instance of the MongoDB RAG.
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
    Get the RAG system for the specified mode and database engine.
    :param mode: The mode of operation (prod, dev, test).
    :param db_engine: The database engine to use (mongodb).
    :param embedding_model: The embedding model to use (google, none, fake).
    :return: An instance of the RAG system.
    """
    db = get_db_client(db_engine)
    embedding_model = get_embedding_model(embedding_model)
    rag_func = globals().get(f"_get_{db_engine}_rag")
    return rag_func(db, mode, embedding_model)

#------------------------------------------------------------

# Get LangGraph checkpointer from config
# name pattern: _get_{checkpointer_mode}_saver

def _get_mongodb_saver(db_uri_key: str, db_name: str):
    """
    Get the MongoDB saver for the specified URI key and database name.
    :param db_uri_key: The environment variable key for the MongoDB URI.
    :param db_name: The name of the database to connect to.
    :return: An instance of the MongoDB saver.
    """
    db_client = MongoClient(os.getenv(db_uri_key))
    return MongoDBSaver(db_client, db_name=db_name)

def _get_local_saver():
    """
    Get the local saver.
    :return: An instance of the local saver.
    """
    return MemorySaver()

def get_checkpointer(checkpointer_mode: Literal['mongodb', 'local']):
    """
    Get the checkpointer based on the specified mode.
    :param checkpointer_mode: The mode of operation (mongodb, local).
    :return: An instance of the checkpointer.
    """
    checkpointer_config = getattr(vars, f"{checkpointer_mode}_saver_config")
    saver_func = globals().get(f"_get_{checkpointer_mode}_saver")
    return saver_func(**checkpointer_config.params())


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
    
    llm = get_llm(llm_model)
    rag = get_rag(mode, db_engine, embedding_model)
    rag_kwargs = getattr(getattr(vars, f"{db_engine}_rag_config"), mode)
    memory = get_checkpointer(chackpointer_mode)
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







