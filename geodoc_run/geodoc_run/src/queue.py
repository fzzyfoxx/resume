from google.cloud import bigquery
from google.auth import default
from geodoc_config import get_service_config, load_config

def get_queue_insert_spec(service_name, source_type, source_key):

    # Get target table info
    service_config = get_service_config(service_name, 'worker')
    queue_table = service_config['queue_table']
    queue_dataset = service_config['target_dataset_id']

    # Get source table info
    source_config = load_config(source_type)
    source_dataset = source_config['dataset_id']
    source_table = source_config['tables'][source_key]

    return {
        'queue_table': queue_table,
        'queue_dataset': queue_dataset,
        'source_dataset': source_dataset,
        'source_table': source_table
    }

def add_teryts_to_queue(service_name, source_type, source_key, teryt_pattern, priority):
    """
    Adds a TERYT code to the processing queue.
    
    Args:
        service_name (str): Name of the service to which the task is added.
        source_type (str): Type of the source data (e.g., 'areas').
        source_key (str): Key for the specific source table.
        teryt_pattern (str): TERYT code pattern to match.
        priority (int): Priority of the task in the queue.
    """
    spec = get_queue_insert_spec(service_name, source_type, source_key)

    client = bigquery.Client()
    project_id = client.project
    query = f"""
        INSERT INTO `{project_id}.{spec['queue_dataset']}.{spec['queue_table']}`
        SELECT source.teryt as teryt, '{spec['source_table']}' as table_name, {priority} as priority
        FROM 
        `{project_id}.{spec['source_dataset']}.{spec['source_table']}` as source
        LEFT JOIN `{project_id}.{spec['queue_dataset']}.{spec['queue_table']}` as target ON source.teryt = target.teryt
        WHERE 
        source.teryt LIKE '{teryt_pattern}%' AND
        target.teryt IS NULL
    """

    result = client.query(query).result()
    print(f"Added {result.num_dml_affected_rows} rows to the queue in {spec['queue_dataset']}.{spec['queue_table']}.")

