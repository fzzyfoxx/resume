from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any, TypedDict, Annotated
from pydantic import BaseModel
from langgraph.constants import Send
import json
import random
from operator import add
from rag_eval.prompt_manager import PromptManager
from rag_eval.utils import create_path_if_not_exists, arxiv_search, download_pdf
import fitz  # PyMuPDF
import os

class arXivDocSpec(TypedDict):
    title: str
    summary: str
    published: str
    url: str

class PapersRetrieveGraphState(BaseModel):
    search_query: str
    docs_specs: List[arXivDocSpec] = []
    metadata_files: Annotated[List[str], add] = []

class PapersRetrieveGraph:
    def __init__(
            self,
            docs_path,
            docs_metadata_path,
            memory=None,
            max_results: int = 5,
    ):
        self.docs_path = docs_path
        self.docs_metadata_path = docs_metadata_path
        self.memory = memory
        self.max_results = max_results

        create_path_if_not_exists(self.docs_path)
        create_path_if_not_exists(self.docs_metadata_path)

        self.build_graph()

    def search_node(self, state: PapersRetrieveGraphState):
        """
        Searches for papers on arXiv based on the search query in the state.
        """
        print(f"Searching for papers with query: {state.search_query}")
        docs_specs = arxiv_search(state.search_query, max_results=self.max_results, start=0)
        return {'docs_specs': docs_specs}
    
    def download_router(self, state: PapersRetrieveGraphState):
        """
        Distributes docs specifications into download nodes.
        """
        return [Send("doc_downloader", doc_spec) for doc_spec in state.docs_specs]
    
    def doc_download_node(self, doc_spec: arXivDocSpec):
        """
        Downloads a PDF file based on the document specification provided and saves its metadata.
        """
        #print(f"Downloading PDF for: {doc_spec['title']}")
        try:
            result = download_pdf(doc_spec['url'], self.docs_path)
            if result is not None:
                doc_spec['path'] = result
                # Get pdf file pages count with PyMuPDF
                try:
                    pdf_document = fitz.open(result)
                    doc_spec['pages_count'] = pdf_document.page_count
                    pdf_document.close()
                except Exception as e:
                    print(f"Error reading PDF file {result}: {e}")
                    doc_spec['pages_count'] = None
                # Save doc_spec to metadata json file
                metadata_file = os.path.join(self.docs_metadata_path, f"{'.'.join(os.path.basename(result).split('.')[:-1])}.json")
                with open(metadata_file, 'w') as f:
                    json.dump(doc_spec, f, indent=4)
                return {'metadata_files': [metadata_file]}
        except Exception as e:
            print(f"Error downloading PDF: {e}")
        return {'metadata_files': []}
    
    def build_graph(self):
        """
        Builds the state graph for retrieving papers from arXiv.
        """
        workflow = StateGraph(PapersRetrieveGraphState)
        workflow.add_node('search_docs', self.search_node)
        workflow.add_node('doc_downloader', self.doc_download_node)

        workflow.add_edge(START, 'search_docs')
        workflow.add_conditional_edges('search_docs', self.download_router, ['doc_downloader'])
        workflow.add_edge('doc_downloader', END)

        self.graph = workflow.compile(checkpointer=self.memory)

    def run(self, search_query: str, thread_id: str = None):
        """
        Runs the graph with the provided search query.

        Args:
            search_query (str): The query to search for papers on arXiv.
        """
        config = {'configurable': {'thread_id': thread_id}} if thread_id else None
        return self.graph.invoke({'search_query': search_query}, config=config)

class QueriesModel(BaseModel):
    queries: List[str]

class KeywordCloudModel(BaseModel):
    keywords: List[str]

class RandomQueriesPaperSearchGraphState(BaseModel):
    queries: QueriesModel = []
    metadata_files: Annotated[List[str], add] = []


class RandomQueriesPaperSearchGraph:
    """
    A graph for generating random queries and searching for papers on arXiv.
    This graph generates random queries, retrieves papers based on those queries,
    and generates additional queries based on the retrieved papers.
    """
    def __init__(
            self,
            llm,
            prompts_config: Dict,
            docs_path: str,
            docs_metadata_path: str,
            memory=None,
            keyword_cloud_size: int = 50,
            main_queries_num: int = 5,
            paper_queries_num: int = 10,
            max_results: int = 5,
            prompt_manager_spec: Dict = {}
    ):
        """
        Initializes the RandomQueriesPaperSearchGraph.
        Args:
            llm: The language model to use for generating queries.
            prompts_config (Dict): Configuration for prompts.
                Configuration should include:
                'path': main path where prompts are stored,
                'random_queries': title of the prompt for generating random queries,
                'paper_queries': title of the prompt for generating queries based on papers.
                'keyword_cloud': title of the prompt for generating keyword cloud.
            docs_path (str): Path to save downloaded documents.
            docs_metadata_path (str): Path to save metadata of downloaded documents.
            memory: Optional memory for the graph.
            keyword_cloud_size (int): Size of the keyword cloud to generate.
            main_queries_num (int): Number of main queries to generate.
            paper_queries_num (int): Number of queries to generate for each paper.
            max_results (int): Maximum number of results to retrieve from arXiv.
            prompt_manager_spec (Dict): Specification for the prompt manager, especially prompts versions.
                ex.
                {'version_config': {'random_queries': 'v1.0', 'paper_queries': 'v1.1'}}
        """
        self.llm = llm
        self.prompts_config = prompts_config
        self.docs_path = docs_path
        self.docs_metadata_path = docs_metadata_path
        self.memory = memory
        self.keyword_cloud_size = keyword_cloud_size
        self.main_queries_num = main_queries_num
        self.paper_queries_num = paper_queries_num
        self.prompt_manager_spec = prompt_manager_spec
        self.max_results = max_results

        create_path_if_not_exists(self.docs_path)
        create_path_if_not_exists(self.docs_metadata_path)

        self._set_promps()
        self._set_downloader()

        self.build_graph()

    def _set_downloader(self):
        """
        Initializes the downloader graph for retrieving papers from arXiv.
        """
        self.downloader = PapersRetrieveGraph(
            docs_path=self.docs_path,
            docs_metadata_path=self.docs_metadata_path,
            max_results=self.max_results
        )

    def _set_promps(self):
        """
        Initializes the prompt manager and retrieves the prompts for generating queries.
        """
        prompt_manager = PromptManager(
            version_config=self.prompt_manager_spec,
            path=self.prompts_config['path']
        )

        self.prompts = prompt_manager.get_prompts([self.prompts_config[name] for name in ['random_queries', 'paper_queries', 'keyword_cloud']])

    def main_queries_node(self, state: RandomQueriesPaperSearchGraphState):
        """
        Generates a set of random queries for searching papers.
        """    
        #print("Generating random queries for paper search...")

        keyword_cloud_prompt = self.prompts['keyword_cloud'].format(
            keyword_cloud_size=self.keyword_cloud_size
        )
        keyword_cloud = self.llm.with_structured_output(KeywordCloudModel).invoke(keyword_cloud_prompt).keywords
        topics = str(random.sample(keyword_cloud, min(self.main_queries_num, len(keyword_cloud))))

        query_prompt = self.prompts['random_queries'].format(
            main_queries_num=self.main_queries_num,
            topics=topics
        )
        queries = self.llm.with_structured_output(QueriesModel).invoke(query_prompt)
        return {'queries': queries}
    
    def query_routing(self, state: RandomQueriesPaperSearchGraphState):
        """
        Distributes queries into paper search nodes.
        """
        return [Send("paper_search", {'search_query': query}) for query in state.queries.queries]
    
    def papers_routing(self, state: RandomQueriesPaperSearchGraphState):
        """
        Distributes metadata files paths into paper queries nodes.
        """
        return [Send("paper_queries", file) for file in state.metadata_files]
    
    def paper_queries(self, metadata_path: str):
        """
        Generates paper queries based on the metadata file.
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        #print(f"Generating queries for paper: {metadata['title']}")
        
        query_prompt = self.prompts['paper_queries'].format(
            title=metadata['title'],
            summary=metadata['summary'],
            paper_queries_num=self.paper_queries_num
        )
        
        queries = self.llm.with_structured_output(QueriesModel).invoke(query_prompt)

        metadata['queries'] = queries.queries
        # Save updated metadata with queries
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        return {}
    
    def build_graph(self):
        """
        Builds the state graph for retrieving papers based on random queries.
        """
        workflow = StateGraph(RandomQueriesPaperSearchGraphState)
        workflow.add_node('main_queries', self.main_queries_node)
        workflow.add_node('paper_search', self.downloader.graph)
        workflow.add_node('paper_queries', self.paper_queries)

        workflow.add_edge(START, 'main_queries')
        workflow.add_conditional_edges('main_queries', self.query_routing, ['paper_search'])
        workflow.add_conditional_edges('paper_search', self.papers_routing, ['paper_queries'])
        workflow.add_edge('paper_queries', END)

        self.graph = workflow.compile(checkpointer=self.memory)

    def run(self, thread_id: str = None):
        """
        Runs the graph to generate random queries and search for papers.

        Args:
            thread_id (str): Optional thread ID for memory management.
        """
        config = {'configurable': {'thread_id': thread_id}} if thread_id else None
        return self.graph.invoke({}, config=config)