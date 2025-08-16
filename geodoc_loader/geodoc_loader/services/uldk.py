import numpy as np
import shapely
import datetime
from shapely import union_all, wkt
from shapely.geometry import Polygon, Point, GeometryCollection
from shapely.prepared import prep
from shapely.validation import make_valid
import re
import asyncio
import aiohttp
import random
import time

from geodoc_loader.handlers.geom import convert_geom_from_wkt
import uuid
from geodoc_loader.handlers.core import list_to_geodataframe, save_geojson, overwrite_metaparams_from_env
from geodoc_loader.download.gcp import upload_geom_set_to_bigquery
from geodoc_loader.area.queries import get_teryt_geom_left_by_teryt_pattern_query, get_query_result


def _force_geometry_collection(geom):
    """
    Force a geometry to be a GeometryCollection if it is not already.
    If the geometry is empty, it will return an empty GeometryCollection.
    Args:
        geom (shapely.geometry): The geometry to check.
    Returns:
        shapely.geometry.GeometryCollection: A GeometryCollection containing the input geometry.
    If the input geometry is empty, it returns an empty GeometryCollection.
    """
    try:
        len(geom.geoms)
    except:
        geom = shapely.geometry.GeometryCollection([geom])
    return geom


def generate_irregular_grid_points(
    search_area: GeometryCollection,
    min_grid_size: float,
    target_points: int
) -> GeometryCollection:
    """
    Generates a GeometryCollection of points within a given search area (GeometryCollection of polygons).

    This function generates points by focusing on each polygon individually, creating a
    local grid within its bounds. The target number of points is distributed proportionally
    to polygon areas. Irregularity is introduced by applying a random row-wise shift
    to the grid points within each polygon.

    Args:
        search_area: A shapely GeometryCollection containing Polygon objects
                     where points should be generated.
        min_grid_size: The minimum desired distance between any two generated grid points.
                       This constraint will be respected, potentially leading to fewer
                       points than 'target_points' if polygons are too small or too few.
        target_points: The approximate total number of grid points desired across all
                       polygons combined.

    Returns:
        A shapely GeometryCollection containing the generated Point objects.
    """
    
    # 1. Filter valid polygons from the search_area
    polygons = [g for g in search_area.geoms if isinstance(g, Polygon) and not g.is_empty]
    
    if not polygons:
        print("Warning: No valid polygons found in the search_area. Returning empty GeometryCollection.")
        return GeometryCollection([])

    # 2. Calculate total area and distribute target points among polygons.
    #    Each polygon gets at least 1 point.
    total_area = sum(p.area for p in polygons)
    if total_area == 0:
        print("Warning: Total area of polygons is zero. Returning empty GeometryCollection.")
        return GeometryCollection([])

    # Use a dictionary to store the target number of points for each polygon.
    points_per_polygon = {}
    
    # Calculate points per unit area for distribution
    points_per_area_unit = target_points / total_area

    # Distribute points, ensuring at least one point per polygon
    for poly_idx, polygon in enumerate(polygons):
        # Calculate ideal points based on area
        num_points = round(polygon.area * points_per_area_unit)
        # Ensure at least 1 point, and handle very small areas that might round to 0
        points_per_polygon[poly_idx] = max(1, num_points)

    #print(f"Distributed target points: {points_per_polygon}")

    # 3. Generate points for each polygon individually.
    final_points = set() # Use a set to automatically handle potential duplicates across polygon boundaries

    for poly_idx, polygon in enumerate(polygons):
        # Calculate effective grid spacing for this specific polygon.
        # This is based on its allocated points, but always respects min_grid_size.
        allocated_points = points_per_polygon[poly_idx]
        
        # Avoid division by zero if for some reason allocated_points becomes 0 (though max(1,..) prevents this)
        if allocated_points == 0:
            continue

        ideal_grid_cell_area_poly = polygon.area / allocated_points
        # The estimated_grid_spacing for this polygon
        estimated_grid_spacing = max(min_grid_size, np.sqrt(ideal_grid_cell_area_poly))
        
        #print(f"Polygon {poly_idx}: Allocated points={allocated_points}, Grid spacing={estimated_grid_spacing:.4f}")

        minx, miny, maxx, maxy = polygon.bounds
        
        # Prepare the current polygon for efficient containment checks
        prepared_polygon = prep(polygon)

        # Generate points on a local grid, applying a row-wise shift
        # Start slightly outside bounds to ensure coverage, as points are then filtered.
        # This starting point can be arbitrary, as long as it allows for full coverage
        # of the polygon's bounding box.
        # We ensure the grid aligns to a "zero" origin and then apply the row shifts.
        current_y = miny + random.uniform(0, estimated_grid_spacing * 0.5)
        while current_y <= maxy + estimated_grid_spacing: # Extend slightly past maxy
            # Apply a random shift for this specific row (y-coordinate line)
            # The shift is applied within [0, estimated_grid_spacing * 0.5) to keep it within reason.
            row_shift_x = random.uniform(0, estimated_grid_spacing * 0.5)
            
            current_x = minx + row_shift_x
            while current_x <= maxx + estimated_grid_spacing: # Extend slightly past maxx
                p = Point(current_x, current_y)
                
                # Check if the point is within the current polygon
                if prepared_polygon.contains(p):
                    final_points.add(p)
                
                current_x += estimated_grid_spacing
            current_y += estimated_grid_spacing
    
    # 4. Final check: Ensure every polygon still has at least one point,
    #    as the grid generation might sometimes miss a point in very specific cases
    #    (e.g., extremely narrow polygons where grid cells are too large).
    #    This serves as a robust fallback.
    for polygon in polygons:
        has_point_in_polygon = False
        for p in final_points:
            if polygon.contains(p):
                has_point_in_polygon = True
                break
        
        if not has_point_in_polygon:
            # If no generated point is inside this polygon, add its representative point.
            # `representative_point()` is guaranteed to be within the polygon.
            #print(f"Adding representative point for a polygon that missed grid points: {polygon.representative_point()}")
            final_points.add(polygon.representative_point())

    #print(f"Total points generated: {len(final_points)}")
    
    # 5. Return the result as a GeometryCollection of points.
    return GeometryCollection(list(final_points))

def filter_by_pattern(items, pattern):
    ans = [x if re.match(pattern, x) else None for x in items]
    return list(filter(lambda x: x, ans))

def parse_uldk_response(resp, split_resp_pattern=r';|\n|\|',id_pattern=r'^\d{6}_\d{1}.\d{4}.', wkt_pattern = r'POLYGON\('):
    """
    Parse the ULDK response to extract IDs and WKT polygons.
    Args:
        resp (bytes): The response from the ULDK service.
        split_resp_pattern (str): Regex pattern to split the response into items.
        id_pattern (str): Regex pattern to match IDs.
        wkt_pattern (str): Regex pattern to match WKT polygons.
    Returns:
        tuple: A tuple containing two lists:
            - List of IDs matching the id_pattern.
            - List of WKT polygons matching the wkt_pattern.
    """
    
    items = re.split(split_resp_pattern, resp.decode())

    ids = filter_by_pattern(items, id_pattern)
    polygons = filter_by_pattern(items, wkt_pattern)

    return ids, polygons

async def get_parcel_from_url_async(url, session, req_timeout):
    try:
        async with session.get(url=url, timeout=req_timeout) as response:
            resp = await response.read()
        try:
            parcel_id, parcel_wkt = parse_uldk_response(resp)
            """if len(parcel_wkt)>1:
                parcel_sp = shapely.geometry.GeometryCollection([wkt.loads(x) for x in parcel_wkt])
            else:
                parcel_sp = wkt.loads(parcel_wkt[0])"""
            #print(parcel_id)
            return (make_valid(wkt.loads(parcel_wkt[0])), parcel_id[0], str(datetime.datetime.now())), None
        except Exception as e:
            #print(type(e))
            #print(e)
            url_exception = (str(e),url, resp.decode('utf-8'))
            return None, url_exception
    except Exception as e:
        #print(url, e)
        url_exception = (str(e),url, None)
        return None, url_exception
    
async def get_responses_for_requests(urls, req_timeout):
        async with aiohttp.ClientSession() as session:
            ret = await asyncio.gather(*[get_parcel_from_url_async(url, session, req_timeout) for url in urls])
        return ret

def _get_unique_parcels(parcels_batch, parcels):
    # filter out empty responses
    parcels_batch = list(filter(lambda x: x!=None, parcels_batch))
    if len(parcels_batch)>0:
        # cut off parcels which were already downloaded
        parcels_batch = [x for x in parcels_batch if x[1] not in parcels[:,1]]
        if len(parcels_batch)>0:
            # extract parcels ids
            parcels_ids = [x[1] for x in parcels_batch]
            # get indices of parcels first appearance
            unique_parcels_idxs = np.unique(parcels_ids, return_index=True)[1]
            # filter unique records
            return np.array(parcels_batch, dtype='object')[unique_parcels_idxs]
        else:
            return np.empty(shape=(0,3), dtype='object')
    else:
        return np.empty(shape=(0,3), dtype='object')
    
def download_iteration(parcels, search_area, min_area_size, target_points, max_targets, min_grid_size, url_form, req_timeout):
    # if search area is single polygon convert it to GeometryCollection so it could be iterated
    search_area = _force_geometry_collection(search_area)

    search_area = shapely.geometry.GeometryCollection([a for a in search_area.geoms if a.area>min_area_size])
    
    # calculate number of grid points for every polygon based on its area
    #geoms_target_points = _calc_target_points(search_area, target_points)
    if search_area.area>0:
        # generate grid points for every polygon
        grid_points = generate_irregular_grid_points(search_area, min_grid_size=min_grid_size, target_points=target_points)

        if grid_points.geom_type=='Point':
            grid_points = shapely.geometry.GeometryCollection([grid_points])

        #prepare urls for set of points
        urls = [url_form.format(lat=a.x,lon=a.y, result='teryt,geom_wkt') for a in grid_points.geoms]
        urls = urls[:max_targets]
        if len(urls)==0:
            return parcels, search_area, np.empty(shape=(0,3), dtype='object'), []
        #get responses
        ret = asyncio.run(get_responses_for_requests(urls, req_timeout))
        parcels_batch = [x[0] for x in ret if x[0] is not None]
        url_exceptions = [x[1] for x in ret if x[1] is not None]
        requests_made = len(urls)
        parcels_found = len(parcels_batch)
        # add new parcels to the collection
        parcels_batch = _get_unique_parcels(parcels_batch, parcels)
        parcels = np.concatenate([parcels, parcels_batch], axis=0)
        
        # calculate remaining area
        if len(parcels_batch)>0:
            parcels_batch_geom = shapely.union_all(parcels_batch[:,0])
            search_area = search_area.difference(parcels_batch_geom)
    else:
        parcels_batch = np.empty(shape=(0,3), dtype='object')
        url_exceptions = []
        requests_made = 0
        parcels_found = 0
            
    return parcels, search_area, parcels_batch, url_exceptions, requests_made, parcels_found

def download_all_parcels_for_teryt(
    teryt,
    teryt_table_id,
    source_dataset_id,
    shapes_table,
    target_dataset_id,
    parcel_id_column_name,
    bq_client,
    project_id,
    meta_params,
    teryt_crs,
    source_crs,
    target_crs,
    url_form
):
    """
    Download all parcels for a given TERYT (Territorial Unit) from ULDK service.
    Args:
        teryt (str): The TERYT code to download parcels for.
        teryt_table_id (str): The BigQuery table ID containing TERYT geometries.
        source_dataset_id (str): The dataset ID for the source TERYT geometries.
        shapes_table (str): The BigQuery table ID where shapes will be uploaded.
        target_dataset_id (str): The dataset ID for the target shapes.
        parcel_id_column_name (str): The name of the column for parcel IDs in the shapes table.
        bq_client: BigQuery client instance.
        meta_params (dict): Metadata parameters for the download process.
        teryt_crs (str): Coordinate Reference System for TERYT geometries.
        source_crs (str): Source CRS for the downloaded parcels.
        target_crs (str): Target CRS for the uploaded parcels.
        url_form (str): URL template for ULDK service requests.
    Returns:
        tuple: A tuple containing:
            - GeoDataFrame of downloaded parcels in the target CRS or empty list if no parcels found or critical errors occured.
            - List of errors encountered during the download process.
            - List of logs for the download process.
    """
    try:
        teryt_geom = get_query_result(
            bq_client,
            get_teryt_geom_left_by_teryt_pattern_query(
                teryt=teryt,
                teryt_table_id=teryt_table_id,
                teryt_dataset_id=source_dataset_id,
                shapes_table_id=shapes_table,
                shapes_dataset_id=target_dataset_id,
                shapes_teryt_column=parcel_id_column_name,
                project_id=project_id
            )
        )[0]
    except Exception as e:
        error_msg = f'Error fetching teryt geometry for {teryt}: {e}'
        print(error_msg)
        errors = [
            {
                'id': str(uuid.uuid4()),
                'teryt': teryt,
                'error_message': error_msg,
                'timestamp': str(datetime.datetime.now())
            }
        ]
        return [], errors, []

    try:
        search_area = convert_geom_from_wkt(
            teryt_geom['geometry'],
            in_crs=teryt_crs,
            out_crs=source_crs
        )
        if search_area.is_empty:
            print(f'Search area is empty for teryt {teryt}. Skipping.')
            return [], [], []
        teryt_area = search_area.area
        if teryt_area < meta_params['min_area_size']:
            print(f'Teryt area {teryt_area} is smaller than minimum area size {meta_params["min_area_size"]}. Skipping.')
            return [], [], []
    except Exception as e:
        error_msg = f'Error converting teryt geometry for {teryt}: {e}'
        print(error_msg)
        errors = [
            {
                'id': str(uuid.uuid4()),
                'teryt': teryt,
                'error_message': error_msg,
                'timestamp': str(datetime.datetime.now())
            }
        ]
        return [], errors, []
    
    max_fails = meta_params['max_fails']
    parcels = np.empty(shape=(0,3), dtype='object')
    total_requests_made = 0
    url_exceptions = []
    fails = 0
    errors = []
    
    print(f'\n' + '-'*50 + '\n')
    start_time = time.time()

    for i in range(meta_params['max_iterations']):
        try:
            if search_area.is_empty | (search_area.area < meta_params['min_area_size']):
                break
            
            parcels, search_area, parcels_batch, new_url_exceptions, new_requests, parcels_found = download_iteration(
                parcels=parcels,
                search_area=search_area,
                min_area_size=meta_params['min_area_size'],
                target_points=meta_params['target_points'],
                max_targets=meta_params['max_targets'],
                min_grid_size=meta_params['min_grid_size'],
                url_form=url_form,
                req_timeout=meta_params['req_timeout']
            )

            new_parcels_num = len(parcels_batch)
            step_success_rate = new_parcels_num / new_requests * 100 if new_requests > 0 else 0
            covered_area = (1-search_area.area/ teryt_area) * 100
            print(f'Step: {i+1} | Covered area: {covered_area:.2f}% | Requests made: {new_requests} | New parcels found: {new_parcels_num} | Success rate: {step_success_rate:.2f}% | Duplicates: {parcels_found - new_parcels_num}')

            total_requests_made += new_requests
            url_exceptions.extend(new_url_exceptions)

            fails = fails + 1 if new_parcels_num == 0 else 0
        except Exception as e:
            error_msg = f'Error during download iteration {i+1} for teryt {teryt}: {e}'
            errors.append({
                'id': str(uuid.uuid4()),
                'teryt': teryt,
                'error_message': error_msg,
                'timestamp': str(datetime.datetime.now())
            })
            print(error_msg)
            fails += 1

        if fails >= max_fails:
            print(f'Max fails reached: {fails}. Stopping iteration.')
            break

    time_taken = round(time.time() - start_time,2)
    parcels_found = len(parcels)
    total_success_rate = parcels_found / total_requests_made * 100 if total_requests_made > 0 else 0

    print(f'\n' + '-'*50 + '\n')
    print(f'Search for teryt {teryt} finished.')
    print(f'Total requests made: {total_requests_made}')
    print(f'Parcels found: {parcels_found}')
    print(f'Covered area: {covered_area:.2f}%')
    print(f'Total success rate: {total_success_rate:.2f}%')
    print(f'Iterations made: {i+1}')
    print(f'Time taken: {time_taken:.2f}s')

    if len(url_exceptions) > 0:
        print('*'* 10)
        print(f'Exceptions during requests: {len(url_exceptions)}')
        for e in url_exceptions[:5]:
            print(e)
        if len(url_exceptions) > 5:
            print(f'... and {len(url_exceptions) - 5} more exceptions')
        print('*'* 10)

    if len(parcels) == 0:
        print(f'No parcels found for teryt {teryt}.')
        return [], errors, []
    
    try:
        parcels_to_upload = [{'geometry': x[0], parcel_id_column_name: x[1], "creation_date": x[2]} for x in parcels if teryt in x[1] and x[1] is not None]
        parcels_gdf = list_to_geodataframe(parcels_to_upload, geometry_col='geometry', crs=source_crs, constant_columns=None).to_crs(target_crs)
        area_left = search_area.area
        logs = [
            {
                'id': str(uuid.uuid4()),
                'teryt': teryt,
                'parcels_found': parcels_found,
                'requests_made': total_requests_made,
                'url_exceptions': len(url_exceptions),
                'covered_area': round(covered_area,2),
                'area_left': round(area_left,2),
                'success_rate': round(total_success_rate,2),
                'time_taken': time_taken,
                'last_edition': str(datetime.datetime.now())
            }
        ]

        return parcels_gdf, errors, logs
    except Exception as e:
        error_msg = f'Error converting parcels to GeoDataFrame for teryt {teryt}: {e}'
        print(error_msg)
        errors.append({
            'id': str(uuid.uuid4()),
            'teryt': teryt,
            'error_message': error_msg,
            'timestamp': str(datetime.datetime.now())
        })
        return [], errors, []
    
def download_and_upload_parcels(
        teryt,
        teryts_table_id,
        project_id,
        bq_client,
        storage_client,
        config,
    ):
    """
    Download and upload parcels for a given TERYT.
    Args:
        teryt (str): The TERYT code to download parcels for.
        teryts_table_id (str): The BigQuery table ID containing TERYT geometries.
        project_id (str): The GCP project ID.
        bq_client: BigQuery client instance.
        storage_client: GCP Storage client instance.
        config (dict): Configuration dictionary containing various parameters.
    Returns:
        tuple: A tuple containing upload success flags for shapes, errors, and logs.
    """

    meta_params = overwrite_metaparams_from_env(config['meta_params'])

    # download and process parcels for the given teryt
    teryt_gdf, errors, logs = download_all_parcels_for_teryt(
        teryt=teryt,
        project_id=project_id,
        teryt_table_id=teryts_table_id,
        source_dataset_id=config['source_dataset_id'],
        shapes_table=config['shapes_table'],
        target_dataset_id=config['target_dataset_id'],
        parcel_id_column_name=config['parcel_id_column_name'],
        bq_client=bq_client,
        meta_params=meta_params,
        teryt_crs=config['teryt_crs'],
        source_crs=config['source_crs'],
        target_crs=config['target_crs'],
        url_form=config['url_form']
    )

    # Upload results to BigQuery
    shapes_upload, error_upload, log_upload = upload_geom_set_to_bigquery(
        gdf=teryt_gdf,
        errors_data=errors,
        log_data=logs,
        temp_folder=config['temp_folder'],
        geojson_filename=f"{config['geojson_prefix']}_{teryt}",
        bq_client=bq_client,
        storage_client=storage_client,
        project_id=project_id,
        bucket_name=config['bucket_name'],
        bucket_folder_name=config['bucket_folder_name'],
        dataset_name=config['target_dataset_id'],
        shapes_table=config['shapes_table'],
        errors_table=config['errors_table'],
        log_table=config['log_table']
    )

    return shapes_upload, error_upload, log_upload