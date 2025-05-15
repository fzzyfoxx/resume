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

class RAGResearcher:
    """
        RAGResearcher class to perform research using external web sources, llm interpretation and RAG storage.

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
    """
    def __init__(self, 
                 llm, 
                 rag_module,
                 rag_kwargs,
                 memory,
                 web_search_client,
                 web_search_kwargs,
                 max_queries_num,
                 prompt_manage_spec={}
                 ):
        
        self.llm = llm

        self.rag_module = rag_module
        self.rag_kwargs = rag_kwargs

        self.web_search_client = web_search_client
        self.web_search_kwargs = web_search_kwargs

        self.memory = memory if memory else MemorySaver()

        self.max_queries_num = max_queries_num

        self.pm = PromptManager(**prompt_manage_spec)
        self._load_prompts()

        self.web_doc_vector_fields = self.rag_module.get_vec_fields(self.rag_kwargs['search_index_name'])
        self.load_web_docs = self.rag_kwargs.get('load_web_docs', False)


    def _load_prompts(self):
        """
        Load the prompts from the specified paths.
        """
        self.prompts = self.pm.get_prompts(['prepare_web_query', 'summarize_web_output', 'web_docs_concat_field'])


    def _set_prepare_queries_func(self):

        def prepare_queries(state: QueriesState) -> QueriesState:

            template_inputs = state.template_inputs | {'current_question': state.current_question, 'max_queries_num': self.max_queries_num}

            queries = self.llm.with_structured_output(QueryListModel).invoke(self.prompts['prepare_web_query'].format(**template_inputs))

            return {'queries': queries.queries}
        
        def queries_routing(state: QueriesState):
            return [Send('web_search', {'current_question': state.current_question, 'query': query}) for query in state.queries]
           # return [Send('web_search', self.WebSearchState(current_question=state.current_question, query=query, urls_response=[])) for query in state.queries]
        
        return prepare_queries, queries_routing
        
    def _set_run_web_search_func(self):

        def run_web_search(state: WebSearchState) -> WebResponseRoutingModel:
            print('run_web_search', state)
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
            print('url_responses_routing', state)
            return [Send('web_output_handler', url_response) for url_response in state.responses]
        
        return run_web_search, urls_responses_routing
    
    def _form_web_document(self, weboutput: WebOutputModel, url: str, query: str, thread_id: str, user_id: str) -> Document:
        """
        Form a document from a web output llm summary
        Args:
            weboutput: WebOutputModel instance with the output from the llm
            url: URL of the web page
            query: Query used to search for the web page
        Returns:
            Document instance with the web page content and metadata
        """
        return weboutput.model_dump() | {'url': url, 'query': query, 'thread_id': thread_id, 'user_id': user_id}


    def _load_document(self, ):
        pass
    
    def _set_web_output_handler_func(self):

        def web_output_handler(state: WebSearchOutputHandlerState, config: RunnableConfig):
            print('web_output_handler', state)
            response = self.llm.with_structured_output(WebOutputModel).invoke(self.prompts['summarize_web_output'].format(current_question=state.current_question, query=state.query, url_content=state.url_content))
            response_doc = self._form_web_document(response, state.url, state.query, config['configurable']['thread_id'], config['configurable']['user_id'])
            
            return {'documents': response_doc}
        
        return web_output_handler
    
    def _format_web_doc_for_answer(self, web_doc: Dict[str, str]) -> str:

        return self.prompts['web_docs_concat_field'].format(**web_doc)
            
    def _set_web_output_aggregation_func(self):

        def web_output_aggregation(state: RAGGeneralState):
            print(state)
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
        This method defines the sequence of operations to be performed during the web search process.
        It includes preparing queries, running web searches, handling web search outputs, and aggregating results.
        """
        prepare_queries, queries_routing = self._set_prepare_queries_func()
        run_web_search, urls_responses_routing = self._set_run_web_search_func()
        web_output_handler = self._set_web_output_handler_func()
        web_output_aggregation = self._set_web_output_aggregation_func()

        web_search_workflow = StateGraph(self.GeneralState)
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
        display(Image(graph.get_graph(xray=1).draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER), height=200, width=200))

    @property
    def show_web_search_graph(self):
        """
        Display the web search graph.
        """
        self.display_graph(self.web_search_graph)