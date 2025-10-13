
def get_queue_items_query(project_id, dataset_id, table_id, limit, desc_priority=True, filter=None, order_by='teryt'):
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
    query = f"""
    DELETE FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})
    """

    return query