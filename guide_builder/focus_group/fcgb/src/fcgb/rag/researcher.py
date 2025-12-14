from operator import add
from langgraph.constants import Send
from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from bson.timestamp import Timestamp
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel
from fcgb.types.rag import QueryListModel, WebOutputModel, RAGGeneralState, QueriesState, WebSearchState, WebSearchOutputHandlerState, WebResponseRoutingModel
from fcgb.prompt_manager import PromptManager
import nest_asyncio

class RAGResearcher:
    """
    RAGResearcher class to perform research using external web sources, LLM interpretation, and RAG storage.

    This class defines a workflow for retrieving and processing information from web sources, 
    summarizing the results, and optionally storing the data in a RAG (Retrieval-Augmented Generation) module.

    Attributes:
        llm: LangChain LLM instance for generating queries and summaries.
        rag_module: RAG module instance (e.g., MongodbRAG) for managing vector search indexes and document storage.
        rag_kwargs: Named arguments for the RAG module methods.
        memory: LangGraph Checkpointer instance for state persistence.
        web_search_client: Web search client instance for performing web searches and extracting content.
        web_search_kwargs: Named arguments for the web search client methods.
        max_queries_num: Maximum number of queries to generate during a single iteration.
        pm: PromptManager instance for managing prompts.
        prompts: Loaded prompts for query preparation, web output summarization, and document formatting.
        web_doc_vector_fields: Vector field specifications for the RAG module.
        load_web_docs: Boolean indicating whether to load web documents into the RAG module.
        web_search_graph: Compiled state graph for the web search workflow.
    """
    def __init__(self, 
                 llm, 
                 rag_module,
                 rag_kwargs,
                 memory,
                 web_search_client,
                 web_search_kwargs,
                 max_queries_num,
                 prompt_manager_spec={}
                 ):
        """
        Initialize the RAGResearcher instance with the given parameters.

        Args:
            llm: LangChain LLM instance
            rag_module: RAG module instance (e.g., MongodbRAG)
            rag_kwargs: named arguments for the rag_module methods:
                e.g.
                {
                    "search_index_name": search index of MongoDB with vector fields,
                    "load_web_docs": if load separate web documents to the RAG storage
                }
            memory: LangGraph Checkpointer instance (e.g., MemorySaver, MongoDBSaver)
            web_search_client: Web search client instance (e.g., TavilyClient from tavily library) 
                Must have a method 'search' that returns a dictionary with 'results' key containing a list of dictionaries with 'url' key:
                {
                    "results": [
                        {"url": "https://example.com"},
                        ...
                    ]
                }
                And a method 'extract' that takes a list of URLs and returns a dictionary with 'results' key containing a list of dictionaries with 'raw_content' and 'url' key:
                {
                    "results": [
                        {"raw_content": "content from url", "url": "https://example.com"},
                        ...
                    ]
                }
            web_search_kwargs: named arguments for the web search_client methods:
            {
                "search": {...},
                "extract": {...}
            }
            prompts_paths: Paths to the prompt files with spec:
                {
                    "prepare_web_query": "path/to/prepare_web_query.txt",
                    "summarize_web_output": "path/to/summarize_web_output.txt",
                    "web_docs_concat_field": "path/to/web_docs_concat_field.txt",
                }
            max_queries_num: Maximum number of queries to generate during single iteration
            prompt_manager_spec: named arguments for the PromptManager initialization
        """
        
        self.llm = llm

        self.rag_module = rag_module
        self.rag_kwargs = rag_kwargs

        self.web_search_client = web_search_client
        self.web_search_kwargs = web_search_kwargs

        self.memory = memory if memory else MemorySaver()

        self.max_queries_num = max_queries_num

        self.pm = PromptManager(**prompt_manager_spec)
        self._load_prompts()

        self.web_doc_vector_fields = self.rag_module.get_vec_fields(self.rag_kwargs['search_index_name'])
        self.load_web_docs = self.rag_kwargs.get('load_web_docs', False)

        self._set_web_search_graph()

    def _load_prompts(self):
        """
        Load the prompts from the specified paths using the PromptManager.
        """
        self.prompts = self.pm.get_prompts(['prepare_web_query', 'summarize_web_output', 'web_docs_concat_field'])

    def _set_prepare_queries_func(self):
        """
        Define the functions for preparing queries and routing them to the web search nodes.

        Returns:
            Tuple[Callable, Callable]: A tuple containing the query preparation function and the routing function.
        """
        def prepare_queries(state: QueriesState) -> QueriesState:

            template_inputs = state.template_inputs | {'current_question': state.current_question, 'max_queries_num': self.max_queries_num}

            queries = self.llm.with_structured_output(QueryListModel).invoke(self.prompts['prepare_web_query'].format(**template_inputs))

            return {'queries': queries.queries}
        
        def queries_routing(state: QueriesState):
            return [Send('web_search', {'current_question': state.current_question, 'query': query}) for query in state.queries]
           # return [Send('web_search', self.WebSearchState(current_question=state.current_question, query=query, urls_response=[])) for query in state.queries]
        
        return prepare_queries, queries_routing
        
    def _set_run_web_search_func(self):
        """
        Define the functions for running web searches and routing the results to the output handler nodes.

        Returns:
            Tuple[Callable, Callable]: A tuple containing the web search function and the routing function.
        """
        def run_web_search(state: WebSearchState) -> WebResponseRoutingModel:
            urls = [elem['url'] for elem in self.web_search_client.search(state['query'], **self.web_search_kwargs.get('search', {}))['results']]
            urls_response = self.web_search_client.extract(urls, **self.web_search_kwargs.get('extract', {}))['results']

            urls_response = [{
                'current_question': state['current_question'],
                'query': state['query'],
                'url': resp['url'],
                'url_content': resp['raw_content']
            } for resp in urls_response]

            return {'responses': urls_response}
        
        def urls_responses_routing(state: WebResponseRoutingModel):
            return [Send('web_output_handler', url_response) for url_response in state.responses]
        
        return run_web_search, urls_responses_routing
    
    def _form_web_document(self, weboutput: WebOutputModel, url: str, query: str, thread_id: str, user_id: str) -> Document:
        """
        Form a document from a web output LLM summary.

        Args:
            weboutput: WebOutputModel instance with the output from the LLM.
            url: URL of the web page.
            query: Query used to search for the web page.
            thread_id: Thread ID for the current conversation.
            user_id: User ID for the current session.

        Returns:
            Document: A document instance with the web page content and metadata.
        """
        return weboutput.model_dump() | {'url': url, 'query': query, 'thread_id': thread_id, 'user_id': user_id}

    def _load_document(self):
        """
        Placeholder method for loading documents into the RAG module.
        """
        pass
    
    def _set_web_output_handler_func(self):
        """
        Define the function for handling web search outputs and summarizing the content.

        Returns:
            Callable: A function that processes web search outputs and generates summaries.
        """
        def web_output_handler(state: WebSearchOutputHandlerState, config: RunnableConfig):
            response = self.llm.with_structured_output(WebOutputModel).invoke(self.prompts['summarize_web_output'].format(current_question=state.current_question, query=state.query, url_content=state.url_content))
            response_doc = self._form_web_document(response, state.url, state.query, config['configurable']['thread_id'], config['configurable']['user_id'])
            
            return {'documents': response_doc}
        
        return web_output_handler
    
    def _format_web_doc_for_answer(self, web_doc: Dict[str, str]) -> str:
        """
        Format a web document for inclusion in the final answer.

        Args:
            web_doc: A dictionary containing the web document content and metadata.

        Returns:
            str: A formatted string representation of the web document.
        """
        return self.prompts['web_docs_concat_field'].format(**web_doc)
            
    def _set_web_output_aggregation_func(self):
        """
        Define the function for aggregating web search outputs and optionally storing them in the RAG module.

        Returns:
            Callable: A function that aggregates web search outputs and prepares them for the final answer.
        """
        def web_output_aggregation(state: RAGGeneralState):
            created_at = Timestamp(int(datetime.now().timestamp()), 1)
            filtered_contents = [doc.model_dump() | {'created_at': created_at} for doc in state.documents if doc.is_relevant]

            if self.load_web_docs:
                self.rag_module.add_documents(
                    documents=filtered_contents,
                    vector_fields=self.web_doc_vector_fields,
                    task_type='RETRIEVAL_DOCUMENT'
                )

            doc_inputs = [self._format_web_doc_for_answer(doc) for doc in filtered_contents]

            return {'retreived_content': doc_inputs, 'documents': '__clear__', 'responses': '__clear__'}
        
        return web_output_aggregation
        
    def _set_web_search_graph(self):
        """
        Set the web search graph for the RAGResearcher class.

        This method defines the sequence of operations to be performed during the web search process, 
        including preparing queries, running web searches, handling web search outputs, and aggregating results.
        """
        prepare_queries, queries_routing = self._set_prepare_queries_func()
        run_web_search, urls_responses_routing = self._set_run_web_search_func()
        web_output_handler = self._set_web_output_handler_func()
        web_output_aggregation = self._set_web_output_aggregation_func()

        web_search_workflow = StateGraph(RAGGeneralState)
        # Prepare set of queries
        web_search_workflow.add_node('prepare_queries', prepare_queries) 
        web_search_workflow.add_edge(START, 'prepare_queries')
        # Distribute queries to web search nodes
        web_search_workflow.add_node('web_search', run_web_search)
        web_search_workflow.add_conditional_edges('prepare_queries', queries_routing, ['web_search'])
        # Distribute url contents to web output handler nodes
        web_search_workflow.add_node('web_output_handler', web_output_handler)
        web_search_workflow.add_conditional_edges('web_search', urls_responses_routing, ['web_output_handler'])
        # Aggregate web output handler results
        web_search_workflow.add_node('web_output_aggregation', web_output_aggregation)
        web_search_workflow.add_edge('web_output_handler', 'web_output_aggregation')
        web_search_workflow.add_edge('web_output_aggregation', END)

        self.web_search_graph = web_search_workflow.compile(checkpointer=self.memory)

    def display_graph(self, graph):
        """
        Display the specified state graph using Mermaid visualization.

        Args:
            graph: The state graph to display.
        """
        nest_asyncio.apply()
        display(Image(graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER), height=200, width=200))

    @property
    def show_web_search_graph(self):
        """
        Display the web search graph.
        """
        self.display_graph(self.web_search_graph)