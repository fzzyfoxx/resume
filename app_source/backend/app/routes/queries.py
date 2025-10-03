from geodoc_app.search.api_handlers import (
    prepare_query_for_app_filters, 
    prepare_geojson, 
    prepare_geojson_with_attributes,
    get_properties_from_qualification,
    filter_actual_filters,
    get_qualification_from_filters
    )
from geodoc_app.search.search_queries import prepare_area_table_for_search_query, get_prepared_results_query
from geodoc_app.search.parcels_results import get_parcels_result_query
from geodoc_config import load_config, load_config_by_path
from geodoc_app.search.utils import get_query_result
from geodoc_app.search.artificial_results import prepare_artificial_query_result, prepare_artificial_target_result, prepare_artificial_result_output
from geodoc_app.export.features import extract_features
from geodoc_app.export.formats import export_to_csv
import uuid
import time
import os

from flask import Blueprint, jsonify, request, current_app, session, Response
import csv
import io
from shapely.geometry import shape as shapely_shape
import json

from flask import Blueprint, jsonify, request, current_app, session

queries_bp = Blueprint('queries', __name__)

# --- NEW: Ensure base structures exist every request ---
@queries_bp.before_request
def ensure_session_structures():
    if 'queries' not in session:
        session['queries'] = {}
    if 'results' not in session:
        session['results'] = {}
    # Mark as modified so Flask writes cookie if needed
    session.modified = True
# --- END NEW ---

target_table_path = 'app.filters'
RESULTS_METADATA = load_config_by_path('app', 'app_results_metadata.json')

PROJECT_ID = load_config("gcp_general")['project_id']
from google.cloud import bigquery
BQ_CLIENT = bigquery.Client()

"""
@queries_bp.before_request
def log_session_id():
    # This will print the session ID before every request to this blueprint
    current_app.logger.debug(f"Request started for {request.path} with Session ID: {session.sid}")
"""

def save_result_metadata(filterStateId, filter_type, query_metacolumns):
    is_dev = current_app.config.get('DEV', True)
    results = session.get('results', {}).copy()

    if not is_dev:
        results[filterStateId] = {
            'query_metacolumns': query_metacolumns,
            'type': filter_type
        }
    else:
        # -- SAVING QUERY RESULTS FOR TEST PURPOSES --
        metadata = RESULTS_METADATA[filter_type]['metadata']
        res_func = prepare_artificial_query_result if filter_type in ['FilterResult', 'SearchArea'] else prepare_artificial_target_result
        artificial_result = res_func({**query_metacolumns, **metadata, 'type': filter_type})
        results[filterStateId] = artificial_result
        # -- END SAVING QUERY RESULTS FOR TEST PURPOSES --

    session['results'] = results
    session.modified = True  # IMPORTANT

def send_query(query):

    is_dev = current_app.config.get('DEV', True)

    if not is_dev:
        query_job = BQ_CLIENT.query(query)
        query_id = query_job.job_id
        print('BQ QUERY ID:', query_id)
    else:
        query_id = str(uuid.uuid4())

    start_time = time.time()
    # Copy–modify–assign pattern to avoid lost updates
    queries = session.get('queries', {}).copy()
    queries[query_id] = {
        'start_time': start_time,
        'status': 'pending'
    }
    session['queries'] = queries
    session.modified = True   # Ensure persistence

    return query_id

@queries_bp.route('/calculate_filters', methods=['POST'])
def calculate_filters_route():
    """
    Endpoint to calculate filters based on provided parameters.
    Returns:
        JSON response with calculated filters.
    """
    print('SESSION ID /calculate_filters:', session.sid)
    print(request.json)
    filters_req = request.json.get('filters', None)
    filterStateId = request.json.get('filterStateId', None)
    stateId = request.json.get('stateId', None)
    name = request.json.get('name', None)

    if not filterStateId or not stateId:
        return jsonify({"error": "filterStateId and stateId are required"}), 400
    if not filters_req:
        return jsonify({"error": "No filters provided"}), 400
    
    filters = filter_actual_filters(filters_req)
    qualification = get_qualification_from_filters(filters_req)

    project_id = load_config("gcp_general")['project_id']
    FILTER_SELECT_COLUMNS = current_app.config.get('FILTER_SELECT_COLUMNS', ['geometry'])
    INTERSECTION_BUFFER = current_app.config.get('INTERSECTION_BUFFER', 1000)
    SIMPLIFY_RATE = current_app.config.get('SIMPLIFY_RATE', 100)

    qualification_option = qualification['values']['option']
    qualification_buffer = float(qualification['values']['value']) if (qualification['values']['value'] != '') and (qualification['values']['value'] is not None) else 0

    query_metacolumns = {
        'option': qualification_option,
        'buffer': qualification_buffer,
        'filterStateId': filterStateId,
        'stateId': stateId,
        'name': name
    }

    filter_query = prepare_query_for_app_filters(
        filters=filters,
        qualification=qualification.get('values', {'value': None}),
        project_id=project_id,
        query_metacolumns=query_metacolumns,
        target_table_path=target_table_path,
        FILTER_SELECT_COLUMNS=FILTER_SELECT_COLUMNS,
        INTERSECTION_BUFFER=INTERSECTION_BUFFER,
        SIMPLIFY_RATE=SIMPLIFY_RATE
    )

    query_id = send_query(filter_query)

    print('-' * 20)
    print('FilterResult')
    print(f"Query ID: {query_id}")
    for filter in filters:
        print(filter)
    print(qualification)
    print('\nQuery:')
    for line in filter_query.split('\n'):
        print(line.strip())
    print('-' * 20)

    save_result_metadata(filterStateId, 'FilterResult', query_metacolumns)

    try:
        return jsonify({"status": 'ok', "query_id": query_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

@queries_bp.route('/set_search_area', methods=['POST'])
def set_search_area_route():

    print('SESSION ID /set_search_area:', session.sid)
    print(request.json)
    filters_req = request.json.get('filters', [])
    filterStateId = request.json.get('filterStateId', None)
    stateId = request.json.get('stateId', None)
    name = request.json.get('name', None)

    if not filterStateId:
        return jsonify({"error": "filterStateId is required"}), 400
    if not filters_req:
        return jsonify({"error": "No filters provided"}), 400
    
    filters = filter_actual_filters(filters_req)
    teryts_spec = filters[0].get('values', None)
    if not teryts_spec:
        return jsonify({"status": "error", "query_id": None}), 400

    SEARCH_AREA_TABLE_NAME = 'search_area'
    SIMPLIFY_RATE = 100
    project_id = load_config("gcp_general")['project_id']

    adm_config = load_config("administration_units")
    teryt_dataset_id = adm_config['dataset_id']

    query_metacolumns = {
    'option': 'SearchArea',
    'buffer': 0,
    'filterStateId': filterStateId,
    'stateId': filterStateId,
    'name': name
    }

    area_table_query = prepare_area_table_for_search_query(
        teryts_spec=teryts_spec,
        target_table_path=target_table_path,
        query_metacolumns=query_metacolumns,
        project_id=project_id,
        dataset_id=teryt_dataset_id,
        simplify_rate=SIMPLIFY_RATE
    )

    query_id = send_query(area_table_query)

    print('-' * 20)
    print('SearchArea')
    print(f"Query ID: {query_id}")
    for filter in teryts_spec:
        print(filter)
    print('\nQuery:')
    for line in area_table_query.split('\n'):
        print(line.strip())
    print('-' * 20)

    save_result_metadata(filterStateId, 'SearchArea', query_metacolumns)

    try:
        return jsonify({"status": 'ok', "query_id": query_id}), 200
    except Exception as e:
        print('ERROR:', str(e))
        return jsonify({"error": str(e)}), 400
    
@queries_bp.route('/set_search_target', methods=['POST'])
def set_search_target_route():

    print('SESSION ID /set_search_target:', session.sid)
    print(request.json)
    filters_req = request.json.get('filters', [])
    filterStateId = request.json.get('filterStateId', None)
    stateId = request.json.get('stateId', None)
    name = request.json.get('name', None)
    allFilterStateIds = request.json.get('allFilterStateIds', [])

    print('\nALL FILTER STATE IDS:', allFilterStateIds, '\n')

    query_metacolumns = {
    'option': 'TargetObject',
    'filterStateId': filterStateId,
    'stateId': stateId,
    'name': name
    }

    SIMPLIFY_RATE = current_app.config.get('SIMPLIFY_RATE', 100)
    results_table_path = current_app.config.get('RESULTS_TABLE_PATH', 'app.parcel_results')
    filters_table_path = current_app.config.get('FILTERS_TABLE_PATH', 'app.filters')
    parcels_table_path = current_app.config.get('PARCELS_TABLE_PATH', 'parcels.parcels')

    results_query = get_parcels_result_query(
            filters_req=filters_req,
            filterStateId=filterStateId,
            allFilterStateIds=allFilterStateIds,
            simplify_rate=SIMPLIFY_RATE,
            results_table_path=results_table_path,
            filters_table_path=filters_table_path,
            parcels_table_path=parcels_table_path
        )
    
    query_id = send_query(results_query)

    print('-' * 20)
    print('ParcelTarget')
    print(f"Query ID: {query_id}")
    print('\nQuery:')
    for line in results_query.split('\n'):
        print(line.strip())
    print('-' * 20)

    save_result_metadata(filterStateId, 'ParcelTarget', query_metacolumns)

    try:
        return jsonify({"status": 'ok', "query_id": query_id}), 200
    except Exception as e:
        print('ERROR:', str(e))
        return jsonify({"error": str(e)}), 400

@queries_bp.route('/check_query_status', methods=['GET'])
def check_query_status_route():
    """
    Endpoint to check the status of a query based on its ID.
    Returns:
        JSON response with the status of the query.
    """
    query_id = request.args.get('query_id', None)
    if not query_id:
        return jsonify({"error": "Query ID is required"}), 400

    queries = session.get('queries', {})
    query_state = queries.get(query_id)

    if query_state:
        start_time = query_state['start_time']
        elapsed_time = time.time() - start_time
        is_dev = current_app.config.get('DEV', True)
        if not is_dev:
            job = BQ_CLIENT.get_job(query_id)
            if job.state == "DONE":
                query_state['status'] = 'completed'
        else:
            if elapsed_time > 2:
                query_state['status'] = 'completed'

        # Reassign (even if unchanged) to keep deterministic writes
        queries[query_id] = query_state
        session['queries'] = queries
        session.modified = True

        #print(f"[check_query_status] Keys={list(queries.keys())} Returning {query_id} -> {query_state['status']}")
        return jsonify({"status": query_state['status']}), 200

    #print(f"[check_query_status] Missing {query_id}. Existing keys: {list(queries.keys())}")
    return jsonify({"error": "Query not found"}), 404


@queries_bp.route('/get_query_result', methods=['POST'])
def get_query_result_route():
    print('SESSION ID /get_query_result:', session.sid)
    print(request.json)
    try:
        is_dev = current_app.config.get('DEV', True)
        if not is_dev:
            filterStateId = request.json.get('filterStateId', None)
            filter_data = session['results'][filterStateId]

            filter_config = RESULTS_METADATA[filter_data['type']]
            results_table_path = filter_config['result_table_path']
            geometry_key = filter_config['geometry_key']
            attributes_keys = filter_config['attributes_keys']
            metadata = filter_config.get('metadata', {})

            query_metacolumns = filter_data.get('query_metacolumns', {})


            qualification = {
                'option': query_metacolumns.get('option'),
                'value': query_metacolumns.get('buffer')
            }
            style, properties = get_properties_from_qualification(qualification)
            print('STYLE', qualification)
            properties = {**properties, **metadata, **query_metacolumns}

            query = get_prepared_results_query(
                    filterStateId=filterStateId, 
                    project_id=PROJECT_ID, 
                    results_table_path=results_table_path)
            
            print('-' * 20)
            print(f"FilterStateId: {filterStateId}")
            print(f"Results Table Path: {results_table_path}")
            print('\nFinal Query:')
            for line in query.split('\n'):
                print(line.strip())
            print('-' * 20)

            gathered_results = get_query_result(
                client=BQ_CLIENT,
                query=query
            )

            geojson = prepare_geojson_with_attributes(
                gathered_results, 
                geometry_key=geometry_key,
                attributes_keys=attributes_keys, 
                additional_attributes=properties)

        else:
            geojson, style = prepare_artificial_result_output(request.json)

        if len(geojson['features']) > 0:
            print(geojson['features'][0]['properties'])

        return jsonify({
            "geojson": geojson,
            "style": style
        }), 200
    except Exception as e:
        print('ERROR:', str(e))
        return jsonify({"error": str(e)}), 500

@queries_bp.route('/download_feature_csv', methods=['POST'])
def download_feature_csv():
    """
    Accepts a single GeoJSON feature (as sent from the map sidebar) and returns a CSV file.
    Body JSON:
    {
      "feature": {
         "type": "Feature",
         "geometry": {...},
         "properties": {...}
      }
    }
    """
    try:
        payload = request.get_json(silent=True) or {}
        feature = payload.get('feature')
        if not feature:
            return jsonify({"error": "Missing feature in request body"}), 400

        geometry = feature.get('geometry', {})
        properties = feature.get('properties', {}) or {}

        rows, columns, filename_base = extract_features(feature, geometry)
        csv_content, headers = export_to_csv(rows, columns, filename_base)

        return Response(csv_content, headers=headers)
    except Exception as e:
        current_app.logger.exception("Failed to export feature CSV")
        return jsonify({"error": str(e)}), 500

