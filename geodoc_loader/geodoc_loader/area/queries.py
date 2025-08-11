def get_teryts_for_teryts_pattern_query(teryts, table_id, project_id, dataset_id="geo"):

    teryt_len = len(teryts[0])

    query = f"""SELECT
    teryt
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    LEFT(teryt, {teryt_len}) IN UNNEST({teryts})"""

    return query

def get_geom_for_teryts_query(teryts, table_id, project_id, dataset_id="geo"):
    query = f"""SELECT
    ST_UNION_AGG(geometry) AS geometry
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})"""

    return query

def get_geom_for_teryts_with_buffer_query(teryts, buffer, table_id, project_id, dataset_id="geo"):

    query = f"""SELECT
    ST_BUFFER(ST_UNION_AGG(geometry), {buffer}) AS geometry
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})"""

    return query

def get_teryts_for_teryts_with_buffer_query(teryts, buffer, reference_table_id, target_table_id, project_id, dataset_id="geo"):

    teryts_area_query = get_geom_for_teryts_with_buffer_query(teryts, buffer, reference_table_id, project_id, dataset_id)

    query = f"""
    WITH teryts_area AS (
    {teryts_area_query}
    )
    SELECT
    t.teryt,
    FROM
    `{project_id}.{dataset_id}.{target_table_id}` as t
    JOIN teryts_area AS r ON ST_INTERSECTS(t.geometry, r.geometry)
    """

    return query

def get_grid_for_teryts(teryts, teryts_table_id, grid_table_id, project_id, dataset_id='geo'):

    teryts_area_query = get_geom_for_teryts_query(teryts, teryts_table_id, project_id, dataset_id)

    query = f"""
    WITH teryts_area AS (
    {teryts_area_query}
    )
    SELECT
    g.id as cell_id,
    g.geometry
    FROM
    `{project_id}.{dataset_id}.{grid_table_id}` AS g
    JOIN teryts_area AS r ON ST_INTERSECTS(g.geometry, r.geometry)
    """ 

    return query

def get_filtered_grid_for_teryts(teryts, teryts_table_id, grid_table_id, project_id, grid_dataset_id, filter_table_id, filter_dataset_id):

    teryts_area_query = get_geom_for_teryts_query(teryts, teryts_table_id, project_id, grid_dataset_id)

    query = f"""
    WITH teryts_area AS (
    {teryts_area_query}
    )
    SELECT
    g.id as cell_id,
    g.geometry
    FROM
    `{project_id}.{grid_dataset_id}.{grid_table_id}` AS g
    JOIN teryts_area AS r ON ST_INTERSECTS(g.geometry, r.geometry)
    LEFT JOIN `{project_id}.{filter_dataset_id}.{filter_table_id}` AS f ON f.cell_id = g.id
    WHERE f.cell_id IS NULL
    """ 

    return query

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