from geodoc_app.search.api_handlers import (
    prepare_query_for_app_filters, 
    prepare_geojson, 
    get_properties_from_qualification,
    filter_actual_filters,
    get_qualification_from_filters
    )
from geodoc_app.search.search_queries import prepare_area_table_for_search_query
from geodoc_config import load_config
import uuid
import time
import os

from flask import Blueprint, jsonify, request, current_app, session

queries_bp = Blueprint('queries', __name__)

target_table_path = 'app.results'

"""
@queries_bp.before_request
def log_session_id():
    # This will print the session ID before every request to this blueprint
    current_app.logger.debug(f"Request started for {request.path} with Session ID: {session.sid}")
"""

def prepare_artificial_query_result(query_metacolumns):

    # Simulate a random record for demonstration purposes
    data_dir = os.path.join(current_app.root_path, 'data')  # Path to the data directory
    file_path = os.path.join(data_dir, 'example_polygons.csv')
    with open(file_path, "r") as file:
        reader = list(csv.DictReader(file))  # Read rows as dictionaries
        random_record = random.choice(reader)  # Select a random row
    return {
        **query_metacolumns,
        'geometry': random_record['geometry']
    }
    

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
    SEARCH_AREA_TABLE_NAME = current_app.config.get('SEARCH_AREA_TABLE_NAME', 'search_area')
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
    
    query_id = str(uuid.uuid4())
    start_time = time.time()

    print('-' * 20)
    print(f"Query ID: {query_id} | Start Time: {start_time}")
    for filter in filters:
        print(filter)
    print(qualification)
    print('\nQuery:')
    for line in filter_query.split('\n'):
        print(line.strip())
    print('-' * 20)


    session['queries'][query_id] = {
        'start_time': start_time,
        'status': 'pending',
        }
    #session.modified = True

    # -- SAVING QUERY RESULTS FOR TEST PURPOSES --
    artificial_result = prepare_artificial_query_result(query_metacolumns)
    session['results'].append(artificial_result)
    # -- END SAVING QUERY RESULTS FOR TEST PURPOSES --

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

    query_id = str(uuid.uuid4())
    start_time = time.time()

    print('-' * 20)
    print(f"Query ID: {query_id} | Start Time: {start_time}")
    for filter in teryts_spec:
        print(filter)
    print('\nQuery:')
    for line in area_table_query.split('\n'):
        print(line.strip())
    print('-' * 20)

    session['queries'][query_id] = {
        'start_time': start_time,
        'status': 'pending'
        }
    #session.modified = True

    # -- SAVING QUERY RESULTS FOR TEST PURPOSES --
    artificial_result = prepare_artificial_query_result(query_metacolumns)
    session['results'].append(artificial_result)
    # -- END SAVING QUERY RESULTS FOR TEST PURPOSES --

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
    #print(request.args)
    #print('SESSION ID /check_query_status:', session.sid)
    query_id = request.args.get('query_id', None)
    
    if not query_id:
        return jsonify({"error": "Query ID is required"}), 400

    query_state = session['queries'].get(query_id, None)
    #print(session['queries'])
    if query_state:
        start_time = query_state['start_time']
        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            query_state['status'] = 'completed'
            #session['queries'][query_id] = query_state
            #session.modified = True
        print(f"Query ID: {query_id} | Status: {query_state['status']} | Elapsed Time: {elapsed_time:.2f} seconds")
        return jsonify({"status": query_state['status']}), 200

    return jsonify({"error": "Query not found"}), 404


import csv
import random
import shapely.wkt

@queries_bp.route('/get_query_result', methods=['POST'])
def get_query_result_route():
    print('SESSION ID /get_query_result:', session.sid)
    print(request.json)
    filterStateId = request.json.get('filterStateId', None)
    try:
        filter_data = [item for item in session['results'] if item.get('filterStateId', None) == filterStateId][0]
        qualification = {
            'option': filter_data.get('option'),
            'value': filter_data.get('buffer')
        }
        style, properties = get_properties_from_qualification(qualification)
        
        name = filter_data.get('name', None)
        if name:
            properties['Źródło'] = name
        
        geometry = shapely.wkt.loads(filter_data.get('geometry'))
        geojson = prepare_geojson([{'geometry': geometry}], additional_attributes=properties)
        return jsonify({
            "geojson": geojson,
            "style": style
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


    