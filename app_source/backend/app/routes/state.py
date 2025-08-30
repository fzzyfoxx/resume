from geodoc_config import load_config
import uuid
import time
import os

from flask import Blueprint, jsonify, request, current_app, session

state_bp = Blueprint('state', __name__)


@state_bp.route('/save_state', methods=['POST'])
def save_state_route():

    """
    Endpoint to save the current state based on provided parameters.
    Returns:
        JSON response with saved state ID.
    """
    #print(request.json)
    #print('SESSION ID /save_state:', session.sid)
    state = request.json.get('state', {})
    if not state:
        return jsonify({"error": "No state provided"}), 400
    print('STATE:', state)
    main_state = state.get('Obszar wyszukiwania')
    if not main_state:
        return jsonify({"error": "No main state provided"}), 400
    
    state_id = main_state[next(iter(main_state))].get('storedStateId', None)
    session['states'][state_id] = state

    try:
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@state_bp.route('/load_state', methods=['GET'])
def load_state_route():
    """
    Returns the last saved state for the current session.
    """

    #print('SESSION ID /load_state:', session.sid)
    if not session.get('states'):
        return jsonify({"error": "No saved states found"}), 404
    
    last_state_id = list(session['states'].keys())[-1]
    state = session['states'].get(last_state_id, None)
    if not state:
        return jsonify({"error": "No state found for the last state ID"}), 404

    try:
        return jsonify(state), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
