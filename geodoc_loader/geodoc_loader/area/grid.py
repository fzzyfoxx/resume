from google.cloud import bigquery
from google.auth import default
import time

def create_grid_table_bigquery(
    grid_cell_size_meters: int = 1000,
    dataset_id: str = "geo",
    prefix: str = "grid_"
) -> str:
    """
    Creates a grid table in BigQuery covering the union of 'geo.province' geometries.

    The grid cells are approximately 'grid_cell_size_meters' x 'grid_cell_size_meters'.
    Each grid cell will have a unique ID.

    Args:
        grid_cell_size_meters (int): The desired size of each grid cell in meters
                                     (e.g., 1000 for 1km x 1km cells).

    Returns:
        str: The full table ID of the created grid table, or an error message.
    """
    client = bigquery.Client()
    project_id = client.project

    # Construct the dynamic table name
    table_name = f"{prefix}{grid_cell_size_meters}"
    full_table_id = f"{project_id}.{dataset_id}.{table_name}"

    # The BigQuery SQL script as a multi-line string
    # Note: The SQL is embedded directly here.
    # The @grid_cell_size_meters parameter will be passed via query_parameters.
    sql_script = f"""
        CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{table_name}` AS
        WITH base_union_polygon AS (
            -- Compute the aggregated polygon directly within the dynamic SQL.
            SELECT ST_UNION_AGG(geometry) AS geometry FROM `{project_id}.geo.province`
        ),
        BoundingBox AS (
            -- Calculate the bounding box of the input polygon using ST_BOUNDINGBOX.
            -- ST_BOUNDINGBOX returns a STRUCT with xmin, ymin, xmax, ymax fields.
            -- We then access these fields directly.
            SELECT
                ST_BOUNDINGBOX(t1.geometry).xmin AS min_lon,
                ST_BOUNDINGBOX(t1.geometry).ymin AS min_lat,
                ST_BOUNDINGBOX(t1.geometry).xmax AS max_lon,
                ST_BOUNDINGBOX(t1.geometry).ymax AS max_lat
            FROM base_union_polygon AS t1
            WHERE NOT ST_IsEmpty(t1.geometry) AND t1.geometry IS NOT NULL -- Ensure geometry is valid
        ),
        GridParameters AS (
            -- Calculate approximate degree increments for the given cell size in meters.
            SELECT
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                (min_lat + max_lat) / 2 AS ref_lat,
                -- Use the passed @grid_cell_size_meters variable directly
                @grid_cell_size_meters / 111190.0 AS deg_lat_increment,
                -- Replaced PI() with its literal value (approx. 3.14159265359)
                @grid_cell_size_meters / (111320.0 * COS(((min_lat + max_lat) / 2) * 3.14159265359 / 180)) AS deg_lon_increment
            FROM BoundingBox
            WHERE
                -- Ensure latitude is within valid range for COS and COS is not zero
                ABS((min_lat + max_lat) / 2) <= 90 AND COS(((min_lat + max_lat) / 2) * 3.14159265359 / 180) != 0
        )
        SELECT
            GENERATE_UUID() AS cell_id, -- Added unique ID for each grid cell
            grid_cells.cell_geometry as geometry,
        FROM
            (
                -- Subquery to generate potential grid cells within the calculated bounding box
                -- and filter them to include only those that intersect with the input_polygon.
                SELECT
                    lon_start,
                    lat_start,
                    (lon_start + gp.deg_lon_increment) AS lon_end,
                    (lat_start + gp.deg_lat_increment) AS lat_end,
                    -- Convert the constructed WKT string into a GEOGRAPHY polygon object.
                    ST_GeogFromText('POLYGON((' || CAST(lon_start AS STRING) || ' ' || CAST(lat_start AS STRING) || ', ' ||
                    CAST(lon_start + gp.deg_lon_increment AS STRING) || ' ' || CAST(lat_start AS STRING) || ', ' ||
                    CAST(lon_start + gp.deg_lon_increment AS STRING) || ' ' || CAST(lat_start + gp.deg_lat_increment AS STRING) || ', ' ||
                    CAST(lon_start AS STRING) || ' ' || CAST(lat_start + gp.deg_lat_increment AS STRING) || ', ' ||
                    CAST(lon_start AS STRING) || ' ' || CAST(lat_start AS STRING) || '))') AS cell_geometry
                FROM
                    GridParameters AS gp,
                    -- Generate longitude starting points for each cell
                    UNNEST(GENERATE_ARRAY(gp.min_lon, gp.max_lon + gp.deg_lon_increment, gp.deg_lon_increment)) AS lon_start,
                    -- Generate latitude starting points for each cell
                    UNNEST(GENERATE_ARRAY(gp.min_lat, gp.max_lat + gp.deg_lat_increment, gp.deg_lat_increment)) AS lat_start
            ) AS grid_cells
        WHERE
            -- Filter the generated cells to include only those that intersect
            -- with the original input polygon (union of provinces).
            ST_Intersects(grid_cells.cell_geometry, (SELECT geometry FROM base_union_polygon))
    """

    # Configure the query job
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "grid_cell_size_meters", "INT64", grid_cell_size_meters
            )
        ]
    )

    try:
        # Start the query job
        query_job = client.query(sql_script, job_config=job_config)

        # Stream progress
        print("Job started. Streaming progress...")
        while query_job.state not in ("DONE", "FAILED"):
            query_job.reload()  # Refresh job state
            print(f"Job state: {query_job.state}")
            if query_job.total_bytes_processed:
                print(f"Bytes processed: {query_job.total_bytes_processed}")
            time.sleep(2)  # Wait before checking again

        # Check final state
        if query_job.errors:
            error_messages = "\n".join([f"Error: {e['message']}" for e in query_job.errors])
            print(f"Query completed with errors:\n{error_messages}")
            return f"Query completed with errors:\n{error_messages}"
        else:
            print(f"Grid table '{full_table_id}' created successfully.")
            return f"Grid table '{full_table_id}' created successfully."

    except Exception as e:
        return f"An error occurred: {e}"