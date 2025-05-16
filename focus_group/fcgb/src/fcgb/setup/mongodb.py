from fcgb.rag.mongodb import MongodbRAG
import os
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_mongodb import MongoDBAtlasVectorSearch


def setup_mongodb():
    """
        This function creates necessary collections and indexes in MongoDB for the RAG system
    """

    DB_URI = os.getenv('MONGO_URI')

    mongo_client = MongoClient(DB_URI)
    db = mongo_client['FocusGroup']

    rag = MongodbRAG(
            database=db,
            collection_name='vec-web-data',
            embedding_model=None
        )
    
    rag.add_index(
        index_name='web-data-index',
        similarity_func='cosine',
        embedding_dimension=768,
        vector_fields=['relevant_content', 'description'],
        filter_fields=['source', 'user_id', 'thread_id']
    )

    rag = MongodbRAG(
            database=db,
            collection_name='dev-vec-web-data',
            embedding_model=None
        )
    
    rag.add_index(
        index_name='dev-web-data-index',
        similarity_func='cosine',
        embedding_dimension=768,
        vector_fields=['relevant_content', 'description'],
        filter_fields=['source', 'user_id', 'thread_id']
    )

    rag = MongodbRAG(
            database=db,
            collection_name='test-vec-web-data',
            embedding_model=None
        )
    
    rag.add_index(
        index_name='test-web-data-index',
        similarity_func='cosine',
        embedding_dimension=768,
        vector_fields=['relevant_content', 'description'],
        filter_fields=['source', 'user_id', 'thread_id']
    )