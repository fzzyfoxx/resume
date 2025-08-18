from geodoc_app.inputs.values import get_filter_spec
from flask import Blueprint, jsonify, request

filters_bp = Blueprint('filters', __name__)

@filters_bp.route('/get_filter_spec', methods=['GET'])
def get_filter_spec_route():
    """
    Endpoint to get filter specification based on provided parameters.
    Returns:
        JSON response with filter specification.
    """
    print(request.args)
    symbols = request.args.getlist('symbols', [])
    #print(f'symbols using getlist: {symbols}')
    
    # Fallback to manual extraction using lists()
    if not symbols:
        symbols = dict(request.args.lists()).get('symbols', [])
        #print("Symbols extracted using lists():", symbols)
    
    name = request.args.get('name', None)

    try:
        filter_spec = get_filter_spec(symbols, name=name)
        return jsonify(filter_spec), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400