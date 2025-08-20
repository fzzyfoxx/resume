from geodoc_app.inputs.values import get_filter_spec, get_column_items_from_symbols
from geodoc_app.inputs.utils import filter_strings_by_search
from flask import Blueprint, jsonify, request, current_app

filters_bp = Blueprint('filters', __name__)

@filters_bp.route('/get_filter_spec', methods=['GET'])
def get_filter_spec_route():
    """
    Endpoint to get filter specification based on provided parameters.
    Returns:
        JSON response with filter specification.
    """
    print(request.args)

    symbols = dict(request.args.lists()).get('symbols', [])
    name = request.args.get('name', None)

    try:
        filter_spec = get_filter_spec(symbols, name=name)
        for item in filter_spec['filters']:
            print(item)
        return jsonify(filter_spec), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@filters_bp.route('/get_filter_search_hints', methods=['GET'])
def get_filter_search_hints_route():
    """
    Endpoint to get search hints for filters based on provided parameters.
    Returns:
        JSON response with search hints.
    """
    print(request.args)

    symbols = dict(request.args.lists()).get('symbols', [])
    value = request.args.get('value', None)

    n = current_app.config.get('SEARCH_FILTER_HINTS_LIMIT', 20)
    threshold = current_app.config.get('SEARCH_FILTER_HINTS_THRESHOLD', 50)
    max_difference = current_app.config.get('SEARCH_FILTER_HINTS_MAX_DIFFERENCE', 20)

    try:
        items = get_column_items_from_symbols(symbols)
        hints = filter_strings_by_search(
            search=value,
            source=items,
            n=n,
            threshold=threshold,
            max_difference=max_difference
        )
        print(hints)
        return jsonify({"items": hints}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@filters_bp.route('/calculate_filters', methods=['POST'])
def calculate_filters_route():
    """
    Endpoint to calculate filters based on provided parameters.
    Returns:
        JSON response with calculated filters.
    """
    filters = request.json.get('filters', [])
    qualification = request.json.get('qualification', None)

    for filter in filters:
        print(filter)

    print(qualification)

    try:
        return jsonify({"status": 'ok'}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400