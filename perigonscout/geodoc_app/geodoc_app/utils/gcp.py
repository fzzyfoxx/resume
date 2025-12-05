from google.cloud import bigquery
from google.auth import default
from geodoc_config import load_config

def get_bq_client():
    """
    Creates and returns a BigQuery client using default application credentials.
    
    Returns:
        bigquery.Client: An authenticated BigQuery client.
    """
    project_id = load_config("gcp_general")['project_id']
    return bigquery.Client(project=project_id)
