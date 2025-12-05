def get_teryts_for_teryts_pattern_query(teryts, table_id, project_id, dataset_id="geo"):
    """
    Get a SQL query to retrieve TERYT codes and names for given TERYT patterns.

    Args:
        teryts (list of str): List of TERYT code patterns.
        table_id (str): The ID of the table to query.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the table. Defaults to "geo".
    Returns:
        str: A SQL query string.
    """
    teryt_len = len(teryts[0])

    query = f"""SELECT
    teryt,
    name
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    LEFT(teryt, {teryt_len}) IN UNNEST({teryts})"""

    return query

def get_teryts_geoms_query(teryts, table_id, project_id, dataset_id="geo"):
    """
    Generates a SQL query to retrieve TERYT codes and their geometries for given TERYT patterns.

    Args:
        teryts (list of str): List of TERYT code patterns.
        table_id (str): The ID of the table to query.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the table. Defaults to "geo".

    Returns:
        str: A SQL query string.
    """
    teryt_len = len(teryts[0])

    query = f"""SELECT
    teryt,
    geometry
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    LEFT(teryt, {teryt_len}) IN UNNEST({teryts})"""

    return query

def get_geom_for_teryts_query(teryts, table_id, project_id, dataset_id="geo"):
    """
    Generates a SQL query to retrieve the unioned geometry for given TERYT codes.

    Args:
        teryts (list of str): List of TERYT codes.
        table_id (str): The ID of the table to query.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the table. Defaults to "geo".
    Returns:
        str: A SQL query string.
    """
    query = f"""SELECT
    ST_UNION_AGG(geometry) AS geometry
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})"""

    return query

def get_geom_for_teryts_with_buffer_query(teryts, buffer, table_id, project_id, dataset_id="geo"):
    """
    Generates a SQL query to retrieve a buffered geometry for given TERYT codes.

    Args:
        teryts (list of str): List of TERYT codes.
        buffer (float): The buffer distance to apply to the geometry.
        table_id (str): The ID of the table to query.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the table. Defaults to "geo".

    Returns:
        str: A SQL query string.
    """
    query = f"""SELECT
    ST_BUFFER(ST_UNION_AGG(geometry), {buffer}) AS geometry
    FROM
    `{project_id}.{dataset_id}.{table_id}`
    WHERE
    teryt IN UNNEST({teryts})"""

    return query

def get_teryts_for_teryts_with_buffer_query(teryts, buffer, reference_table_id, target_table_id, project_id, dataset_id="geo"):
    """
    Generates a SQL query to retrieve TERYT codes from a target table that intersect with 
    a buffered geometry derived from a reference table.

    Args:
        teryts (list of str): List of TERYT codes.
        buffer (float): The buffer distance to apply to the geometry.
        reference_table_id (str): The ID of the reference table.
        target_table_id (str): The ID of the target table.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the tables. Defaults to "geo".

    Returns:
        str: A SQL query string.
    """
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
    """
    Generates a SQL query to retrieve grid cells that intersect with the geometry of given TERYT codes.

    Args:
        teryts (list of str): List of TERYT codes.
        teryts_table_id (str): The ID of the table containing TERYT geometries.
        grid_table_id (str): The ID of the grid table.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str, optional): The ID of the dataset containing the tables. Defaults to "geo".

    Returns:
        str: A SQL query string.
    """
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
    """
    Generates a SQL query to retrieve grid cells that intersect with the geometry of given TERYT codes,
    excluding cells that are present in a filter table.

    Args:
        teryts (list of str): List of TERYT codes.
        teryts_table_id (str): The ID of the table containing TERYT geometries.
        grid_table_id (str): The ID of the grid table.
        project_id (str): The ID of the Google Cloud project.
        grid_dataset_id (str): The ID of the dataset containing the grid table.
        filter_table_id (str): The ID of the filter table.
        filter_dataset_id (str): The ID of the dataset containing the filter table.

    Returns:
        str: A SQL query string.
    """
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

def get_results_for_teryt_query(teryt, table_id, dataset_id, project_id):
    """
    Generates a SQL query to retrieve results for a specific TERYT from a given table.
    
    Args:
        teryt (str): The TERYT code to filter by.
        table_id (str): The ID of the table to query.
        dataset_id (str): The ID of the dataset containing the table.
        project_id (str): The ID of the Google Cloud project.

    Returns:
        str: A SQL query string to retrieve results for the specified TERYT.
    """
    return f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE teryt = '{teryt}'
    """

def get_teryt_geom_left_by_teryt_pattern_query(teryt, teryt_table_id, teryt_dataset_id, shapes_table_id, shapes_dataset_id, shapes_teryt_column, project_id):
    """
    Generates a SQL query to retrieve the geometry of a TERYT code, subtracting overlapping shapes 
    from another table, and the count of those shapes.

    Args:
        teryt (str): The TERYT code pattern to filter by.
        teryt_table_id (str): The ID of the table containing TERYT geometries.
        teryt_dataset_id (str): The ID of the dataset containing the TERYT table.
        shapes_table_id (str): The ID of the table containing shapes to subtract.
        shapes_dataset_id (str): The ID of the dataset containing the shapes table.
        shapes_teryt_column (str): The column in the shapes table containing TERYT codes.
        project_id (str): The ID of the Google Cloud project.

    Returns:
        str: A SQL query string.
    """
    query = f"""
    WITH shapes AS (
        SELECT ST_UNION_AGG(geometry) as fill, COUNT(*) as shapes_count
        FROM `{project_id}.{shapes_dataset_id}.{shapes_table_id}`
        WHERE {shapes_teryt_column} LIKE '{teryt}%'
        ),
    teryt AS (
        SELECT t.teryt, t.geometry
        FROM `{project_id}.{teryt_dataset_id}.{teryt_table_id}` AS t
        WHERE t.teryt LIKE '{teryt}'
    )
    SELECT 
    teryt.teryt, 
    CASE WHEN shapes.fill IS null THEN teryt.geometry ELSE ST_DIFFERENCE(teryt.geometry, shapes.fill) END AS geometry, 
    shapes.shapes_count
    FROM teryt, shapes
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