from pymongo.operations import SearchIndexModel
from typing import List, Dict, Any, Literal
import warnings

# Customize the warning format
def custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__}: {message}\n"

warnings.formatwarning = custom_warning_format

class MongodbRAG:
    """
    A class for managing MongoDB-based Retrieval-Augmented Generation (RAG) workflows.
    This class provides methods for creating vector search indexes, adding documents with embeddings,
    and retrieving documents based on vector similarity.

    Attributes:
        database: The MongoDB database instance.
        collection_name: The name of the collection used for storing documents.
        embedding_model: The embedding model used for generating vector representations.
        collection: The MongoDB collection object for storing documents.
        index_metadata_collection: The MongoDB collection object for storing index metadata.
    """

    def __init__(self, 
                 database,
                 collection_name,
                 embedding_model,
                 ):
        """
        Initialize the MongodbRAG instance with the given parameters.

        Args:
            database: The MongoDB database instance.
            collection_name: The name of the collection used for storing documents.
            embedding_model: The embedding model used for generating vector representations.
        """
        
        self.database = database
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        if len(self.database.list_collection_names(filter={'name': collection_name}))==0:
            self.database.create_collection(collection_name)
            

        self.collection = self.database[self.collection_name]

        self.index_metadata_collection = self.database['index_metadata']

    def _get_index_definition(self, similarity_func, embedding_dimension, vector_fields, filter_fields):
        """
        Generate the index definition for creating a vector search index.

        Args:
            similarity_func: The similarity function to use ('cosine', 'dot_product', or 'euclidean').
            embedding_dimension: The dimensionality of the vector embeddings.
            vector_fields: A list of vector field specifications.
            filter_fields: A list of filter field specifications.

        Returns:
            dict: The index definition for the vector search index.
        """

        vector_fields = [{
            "type": "vector",
            "numDimensions": embedding_dimension,
            "path": field['vec_path'],
            "similarity": similarity_func,
        } for field in vector_fields]

        filter_fields = [{
            "type": "filter",
            "path": field
        } for field in filter_fields]

        return {"fields": vector_fields + filter_fields}
    
    def _assign_vector_field_name(self, field_name: str):
        """
        Assign a vector field name for a given source field.

        Args:
            field_name (str): The name of the source field.

        Returns:
            str: The name of the corresponding vector field.
        """
        return f"{field_name}_vector"

    def _assign_vector_field_names(self, field_names: List[str]):
        """
        Assign vector field names for a list of source fields.

        Args:
            field_names (List[str]): A list of source field names.

        Returns:
            List[Dict[str, str]]: A list of dictionaries mapping source fields to vector fields.
        """
        return [{'source_path': field, 'vec_path': self._assign_vector_field_name(field)} for field in field_names]

    def add_index(
            self, 
            index_name: str,
            similarity_func: Literal['cosine', 'dot_product', 'euclidean'],
            embedding_dimension: int,
            vector_fields: List[str],
            filter_fields: List[str]
              ):
        """
        Add a vector search index to the MongoDB collection.

        Args:
            index_name (str): The name of the index.
            similarity_func (Literal): The similarity function to use ('cosine', 'dot_product', or 'euclidean').
            embedding_dimension (int): The dimensionality of the vector embeddings.
            vector_fields (List[str]): A list of source fields to be used as vector fields.
            filter_fields (List[str]): A list of fields to be used as filters.

        Returns:
            str: A message indicating whether the index was created or already exists.
        """
        
        vector_field_names = self._assign_vector_field_names(vector_fields)
        index_definition = self._get_index_definition(similarity_func, embedding_dimension, vector_field_names, filter_fields)

        # check if the index already exists
        existing_indexes = self.collection.list_search_indexes()
        if any([index['name'] == index_name for index in existing_indexes]):
            msg = f"Index with name {index_name} already exists in collection {self.collection_name}. Please choose a different name or drop existing index with .drop_index(index_name) method."
            warnings.warn(msg, category=UserWarning)
            return 'index_exists'
        else:
            search_index_model = SearchIndexModel(
                                    definition=index_definition,
                                    name=index_name,
                                    type="vectorSearch"
                                )
            
            self.collection.create_search_index(search_index_model)

            index_metadata = {
                "collection": self.collection_name,
                "index_name": index_name,
                "similarity_func": similarity_func,
                "embedding_dimension": embedding_dimension,
                "vector_fields": vector_field_names,
                "filter_fields": filter_fields
            }

            self.index_metadata_collection.insert_one(index_metadata)
            print(f"Index {index_name} created in collection {self.collection_name}.")
            return 'index_created'

    def drop_index(self, index_name: str):
        """
        Drop a vector search index from the MongoDB collection.

        Args:
            index_name (str): The name of the index to drop.
        """
        # Drop the index from the collection and remove its metadata
        self.collection.drop_search_index(index_name)
        self.index_metadata_collection.delete_one({"index_name": index_name, "collection": self.collection_name})
        print(f"Index {index_name} dropped from collection {self.collection_name}.")

    def get_vec_fields(self, index_name: str):
        """
        Get the vector fields specification for a given index name.

        Args:
            index_name (str): The name of the search index.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the source and vector field names.
        """
        index_metadata = self.index_metadata_collection.find_one({"index_name": index_name, "collection": self.collection_name})

        return index_metadata['vector_fields']
    
    def _get_vec_fields_filter(self, index_name: str):
        """
        Get the vector fields filter for a given index name to exclude vector fields in the $project stage.

        Args:
            index_name (str): The name of the search index.

        Returns:
            dict: A dictionary mapping vector field names to 0 for exclusion.
        """
        return {vec_field['vec_path']: 0 for vec_field in self.get_vec_fields(index_name)}
    
    def add_documents(self, 
                      documents: List[Dict[str, Any]], 
                      vector_fields: List[Dict[str, str]], 
                      task_type: Literal['RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'SEMANTIC_SIMILARITY', 'CLASSIFICATION', 'CLUSTERING'] = 'RETRIEVAL_DOCUMENT'):
        """
        Add documents to the MongoDB collection and generate vector embeddings for specified fields.

        Args:
            documents (List[Dict[str, Any]]): A list of documents to be added.
            vector_fields (List[Dict[str, str]]): A list of dictionaries mapping source fields to vector fields.
            task_type (Literal): The type of embedding task to perform. Options are:
                - 'RETRIEVAL_QUERY'
                - 'RETRIEVAL_DOCUMENT'
                - 'SEMANTIC_SIMILARITY'
                - 'CLASSIFICATION'
                - 'CLUSTERING'
        """
        for field_spec in vector_fields:
            source_field = field_spec['source_path']
            vec_field = field_spec['vec_path']
            embeddings = self.embedding_model.embed_documents([doc[source_field] for doc in documents], batch_size=len(documents), embeddings_task_type=task_type)
            for doc, embedding in zip(documents, embeddings):
                doc[vec_field] = embedding

        self.collection.insert_many(documents)

    def retreive(self, query: str, index_name: str, source_field: str, limit: int, threshold: float, filters: Dict[str, Any]):
        """
        Retrieve documents from the MongoDB collection based on a query and specified parameters.

        This function performs a vector search on the specified index using the query string. It generates a query vector
        using the embedding model and retrieves documents that match the query based on the similarity function defined
        in the index. Additional filters and thresholds can be applied to refine the search results.

        Consecutive Phases:
        1. Generate Query Vector:
           - The query string is converted into a vector representation using the embedding model.
        2. Initialize Filters:
           - If filters are provided, they are added as a `$match` stage in the aggregation pipeline.
        3. Perform Vector Search:
           - A `$vectorSearch` stage is added to the pipeline to perform the vector similarity search.
        4. Apply Threshold Filter:
           - If a threshold is specified, a `$match` stage is added to filter results based on the similarity score.
        5. Execute Aggregation Pipeline:
           - The aggregation pipeline is executed on the MongoDB collection to retrieve the results.

        Args:
            query (str): The query string to search for.
            index_name (str): The name of the search index to use for the vector search.
            source_field (str): The name of the source field in the documents to generate the query vector.
            limit (int): The maximum number of results to return.
            threshold (float): The minimum similarity score required for a document to be included in the results.
            filters (Dict[str, Any]): Additional filters to apply to the search results.

        Returns:
            List[Dict[str, Any]]: A list of documents that match the query and specified parameters.
        """
        vector_field_name = self._assign_vector_field_name(source_field)
        query_vector = self.embedding_model.embed_query(query, embeddings_task_type='RETRIEVAL_QUERY')
        filters = filters or {}

        search_stage = [{
            '$vectorSearch': {
                'index': index_name,
                'path': vector_field_name,
                'queryVector': query_vector,
                'exact': True,
                'filter': filters,
                "limit": limit
            }
        }]

        output_stage = [{
                    '$project': {'score': {'$meta': 'vectorSearchScore'}} | self._get_vec_fields_filter(index_name)
                }]

        threshold_filter = [{'$match': {"score": {"$gt": threshold}}}] if threshold else []

        pipeline = search_stage + output_stage + threshold_filter

        result = self.collection.aggregate(pipeline)

        return list(result)