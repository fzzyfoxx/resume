from fcgb.rag.mongodb import MongodbRAG
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from fcgb.cfg.precompiled import get_rag
from fcgb.cfg.vars import mongodb_rag_config


def setup_mongodb():
    """
        This function creates necessary collections and indexes in MongoDB for the RAG system
    """

    for mode in ['prod', 'dev', 'test']:

        rag = get_rag(
            mode=mode,
            db_engine='mongodb',
            embedding_model='none'
        )

        mode_kwargs = getattr(mongodb_rag_config, mode)
        index_name = mode_kwargs.get('search_index_name')
        embedding_size = mode_kwargs.get('embdding_size')
        similarity_func = mode_kwargs.get('similarity_func')
        vector_fields = mode_kwargs.get('vector_fields')
        filter_fields = mode_kwargs.get('filter_fields')
    
        rag.add_index(
            index_name=index_name,
            similarity_func=similarity_func,
            embedding_dimension=embedding_size,
            vector_fields=vector_fields,
            filter_fields=filter_fields
        )