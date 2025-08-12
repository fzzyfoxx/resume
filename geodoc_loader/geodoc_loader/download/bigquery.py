

def get_query_result(client, query):
    """
    Executes a SQL query using the provided BigQuery client and returns the result.
    Args:
        client (bigquery.Client): The BigQuery client to use for executing the query.
        query (str): The SQL query to execute.
    Returns:
        list: A list of dictionaries representing the rows returned by the query.
    """
    query_job = client.query(query)
    results = query_job.result()
    return [dict(row) for row in results]

def run_delete_query(client, query):
    """
    Executes a DELETE SQL query using the provided BigQuery client.
    Args:
        client (bigquery.Client): The BigQuery client to use for executing the query.
        query (str): The SQL DELETE query to execute.
    Returns:
        int: The number of rows affected by the DELETE operation.
    """
    query_job = client.query(query)
    result = query_job.result()
    return result.num_dml_affected_rows
