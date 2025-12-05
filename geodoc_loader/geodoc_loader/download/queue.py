
def get_queue_items_query(project_id, dataset_id, table_id, limit, desc_priority=True, filter=None, order_by='teryt'):
    """
    Generates a SQL query to retrieve items from the queue table with optional filtering and ordering.
    Args:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        limit (int): The maximum number of items to retrieve.
        desc_priority (bool): Whether to sort by priority in descending order. Defaults to True.
        filter (str): An optional SQL filter clause (without the WHERE keyword).
        order_by (str): An optional column name to order the results by, in addition to priority.
    Returns:
        str: The generated SQL query.
    """
    
    sorting = "DESC" if desc_priority else "ASC"

    order_by = ', '.join(['priority ' + sorting, order_by]) if order_by else 'priority ' + sorting

    filter_clause = f"\nWHERE \n {filter} \n" if filter else ""

    query = f"""
    SELECT
    *
    FROM
    `{project_id}.{dataset_id}.{table_id}`{filter_clause}
    ORDER BY
    {order_by}
    LIMIT {limit}
    """

    return query

def delete_teryt_queue_items_query(project_id, dataset_id, table_id, teryts):
    """
    Generates a SQL query to delete items from the queue table for specified TERYTs.
    Args:
        project_id (str): The GCP project ID.
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table ID.
        teryts (list): A list of TERYT codes to delete from the queue.
    Returns:
        str: The generated SQL query.
    """
    query = f"""
    DELETE FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})
    """

    return query