
def get_queue_items_query(project_id, dataset_id, table_id, limit):
    query = f"""
    SELECT
    *
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    ORDER BY
    priority DESC
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