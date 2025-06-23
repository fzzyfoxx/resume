import os
import requests
from urllib.parse import urlparse
import urllib.request as libreq
import feedparser
import re
from typing import List, Dict, Any
import json

def sanitize_query(query: str) -> str:
    """
    Sanitizes a query string for API calls by removing special characters
    and replacing white spaces with '+'.

    Args:
        query (str): The input query string.

    Returns:
        str: The sanitized query string.
    """
    # Remove special characters except alphanumeric and spaces
    sanitized = re.sub(r'[^\w\s]', '', query)
    # Replace white spaces with '+'
    sanitized = sanitized.replace(' ', '+')
    return sanitized

def create_path_if_not_exists(path: str):
    """
    Creates a directory path if it doesn't exist.

    Args:
        path (str): The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def arxiv_pdf_link_extractor(links: List[Dict[str, str]]) -> str | None:
    """
    Extracts the PDF link from a list of links.
    Args:
        links (List[Dict[str, str]]): A list of dictionaries containing link information.
    Returns:
        str | None: The PDF link if found, otherwise None.
    """

    link = [link['href'] for link in links if link.get('title', '') == 'pdf']
    if len(link) > 0:
        return link[0]
    return None

def arxiv_search(search_query: str, start: int = 0, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the arXiv API for papers matching the search query.

    Args:
        search_query (str): The query to search for.
        start (int): The starting index for results.
        max_results (int): The maximum number of results to return.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing paper titles, summaries, publish dates and URLs.
    """
    api_call = 'http://export.arxiv.org/api/query?search_query=all:%s&start=%i&max_results=%i' % (
        sanitize_query(search_query), start, max_results)
    try:
        response = libreq.urlopen(api_call)
        if response.status != 200:
            raise Exception(f"Error fetching data from arXiv API: {response.status}")
        results = feedparser.parse(response.read())
        docs = [{
            'title': entry.title, 
            'summary': entry.summary,
            'published': entry.published,
            'url': arxiv_pdf_link_extractor(entry.links)
            } for entry in results.entries]
        return [doc for doc in docs if doc['url'] is not None]
    except Exception as e:
        print(e)
        return []
    

def download_pdf(url: str, save_dir: str):
    """
    Downloads a PDF file from a URL to a specified path.

    Args:
        url (str): The URL of the PDF file.
        save_path (str): The path where the PDF file will be saved.
    Returns:
        str: The path where the PDF file was saved.
    """
    # Extract the filename from the URL
    filename = os.path.basename(urlparse(url).path) + '.pdf'
    save_path = os.path.join(save_dir, filename)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        return save_path
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None
    
def sort_dicts_by_id(dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries by the 'id' key.

    Args:
        dicts (List[Dict[str, Any]]): List of dictionaries to sort.

    Returns:
        List[Dict[str, Any]]: Sorted list of dictionaries.
    """
    return sorted(dicts, key=lambda x: x.get('id', float('inf')))


def get_filename(file):
    return '.'.join(os.path.basename(file).split('.')[:-1])

def get_files(path):
    """
    Return list of files from a given folder.
    """
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def get_filenames_list(path):
    """
    Return list of files from a given folder without extensions.
    """
    return [get_filename(f) for f in get_files(path)]

def get_files_without(path, filenames):
    """
    Returns a list of files from a given folder that are not in the provided filenames list.
    """
    return [f for f in get_files(path) if get_filename(f) not in filenames]

def get_filenames_without(path, filenames):
    """
    Returns a list of files from a given folder without extensions that are not in the provided filenames list.
    """
    return [get_filename(f) for f in get_files_without(path, filenames)]

def remove_files(path, filenames):
    """
    Remove files from a given folder that are in the provided filenames list.
    """
    for f in filenames:
        os.remove(os.path.join(path, f))

def load_json(path):
    """
    Load a JSON file from the given path.
    """
    with open(path, 'r') as f:
        return json.load(f)