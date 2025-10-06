import sys
print("IMPORT_LOG: Loading filters_routes.py", file=sys.stderr); sys.stderr.flush()

print("IMPORT_LOG: filters_routes.py - Importing Flask...", file=sys.stderr); sys.stderr.flush()
from flask import Blueprint, jsonify, request, current_app, session
print("IMPORT_LOG: filters_routes.py - DONE", file=sys.stderr); sys.stderr.flush()

print("IMPORT_LOG: filters_routes.py - Creating Blueprint...", file=sys.stderr); sys.stderr.flush()
filters_bp = Blueprint('filters', __name__)
print("IMPORT_LOG: filters_routes.py - DONE", file=sys.stderr); sys.stderr.flush()


@filters_bp.route('/get_filter_spec', methods=['GET'])
def get_filter_spec_route():
    """
    Endpoint to get filter specification based on provided parameters.
    Returns:
        JSON response with filter specification.
    """
    # Lazily import heavy deps
    from geodoc_app.inputs.values import get_filter_spec
    print(request.args)
    print('SESSION ID /get_gilter_spec:', getattr(session, "sid", None))

    symbols = dict(request.args.lists()).get('symbols', [])
    name = request.args.get('name', None)

    try:
        filter_spec = get_filter_spec(symbols, name=name)
        #for item in filter_spec['filters']:
        #    print(item)
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
    # Lazily import heavy deps
    from geodoc_app.inputs.values import get_column_items_from_symbols
    from geodoc_app.inputs.utils import filter_strings_by_search, filter_strings_set_by_search
    #print(request.args)
    #print('SESSION ID /get_filter_search_hints:', session.sid)

    symbols = dict(request.args.lists()).get('symbols', [])
    value = request.args.get('value', None)
    #print(symbols, value)

    n = current_app.config.get('SEARCH_FILTER_HINTS_LIMIT', 20)
    threshold = current_app.config.get('SEARCH_FILTER_HINTS_THRESHOLD', 50)
    max_difference = current_app.config.get('SEARCH_FILTER_HINTS_MAX_DIFFERENCE', 20)

    try:
        items_spec = get_column_items_from_symbols(symbols)
        items = items_spec.get('values', [])
        search_keys = items_spec.get('search_keys', None)
        #print('ITEMS:', items[:3], 'SEARCH KEYS:', search_keys)
        if search_keys is None:
            hints = filter_strings_by_search(
                search=value,
                source=items,
                n=n,
                threshold=threshold,
                max_difference=max_difference
            )
        else:
            hints = filter_strings_set_by_search(
                search=value,
                source=items,
                search_keys=search_keys,
                n=n,
                threshold=threshold,
                max_difference=max_difference
            )
        #print('HINTS:', hints)
        return jsonify({"items": hints}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

print("IMPORT_LOG: Finished loading filters_routes.py", file=sys.stderr); sys.stderr.flush()
