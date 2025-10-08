import sys
#print("IMPORT_LOG: Loading state.py", file=sys.stderr); sys.stderr.flush()

import uuid
import time
import os

from flask import Blueprint, jsonify, request, current_app, session
from datetime import datetime

state_bp = Blueprint('state', __name__)

@state_bp.before_request
def ensure_states_bucket():
    # Ensure the 'states' bucket exists only for state routes
    if 'states' not in session:
        session['states'] = {}
        session.modified = True

@state_bp.route('/save_state', methods=['POST'])
def save_state_route():
    """
    Endpoint to save the current state with a project name.
    Returns:
        JSON response with saved state ID.
    """
    state = request.json.get('state', {})
    project_name = request.json.get('projectName', None)  # Get project name from request
    project_id = request.json.get('projectId') # Get project ID from request or generate a new one
    now = datetime.now().isoformat()
    
    if not project_id:
        project_id = str(uuid.uuid4())
        created_at = now
    else:
        # Existing project, preserve created_at
        existing_data = session.get('states', {}).get(project_id)
        if existing_data and 'metadata' in existing_data and 'created_at' in existing_data['metadata']:
            created_at = existing_data['metadata']['created_at']
        else:
            # Handle case where created_at is missing (e.g., old data)
            created_at = now  # Or a default value like None
    
    if not state:
        return jsonify({"error": "No state provided"}), 400

    if not project_name:
        return jsonify({"error": "No project name provided"}), 400
    
    states = session.get('states', {})
    states[project_id] = {
        'state': state,
        'metadata': {
            'name': project_name,
            'id': project_id,
            'created_at': created_at,
            'last_edition': now
        }
    }
    session['states'] = states
    session.modified = True

    try:
        return jsonify({"status": "ok", "projectId": project_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@state_bp.route('/list_states', methods=['GET'])
def list_states_route():
    """
    Returns a list of saved projects with their names, IDs, dates, and a summary of filters.
    """
    if not session.get('states'):
        return jsonify([]), 200

    projects = []
    for project_id, data in session.get('states', {}).items():
        metadata = data.get('metadata')
        if not metadata:
            continue

        state = data.get('state', {})
        filters_section = state.get('Filtry', {})
        filters_summary = []
        if isinstance(filters_section, dict):
            for chain_data in filters_section.values():
                if isinstance(chain_data, dict) and 'title' in chain_data:
                    filters_summary.append(chain_data['title'])

        project_info = {
            'id': metadata.get('id'),
            'name': metadata.get('name'),
            'creation_date': metadata.get('created_at'),
            'last_edition_date': metadata.get('last_edition'),
            'filters_summary': filters_summary
        }
        projects.append(project_info)
        
    return jsonify(projects), 200

@state_bp.route('/load_state', methods=['GET'])
def load_state_route():
    """
    Returns the saved state for a given project ID.
    """
    project_id = request.args.get('projectId')
    if not project_id:
        return jsonify({"error": "No project ID provided"}), 400

    state_data = session.get('states', {}).get(project_id, None)
    if not state_data:
        return jsonify({"error": "No state found for the given project ID"}), 404

    try:
        return jsonify(state_data['state']), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@state_bp.route('/delete_state', methods=['DELETE'])
def delete_state_route():
    """
    Deletes the saved state for a given project ID.
    """
    project_id = request.args.get('projectId')
    print(f"Received request to delete state for project ID: {project_id}")
    if not project_id:
        return jsonify({"error": "No project ID provided"}), 400
    states = session.get('states', {})
    if project_id in states:
        del states[project_id]
        session['states'] = states
        session.modified = True
        print(f"Deleted state for project ID: {project_id}")
        return jsonify({"status": "ok"}), 200
    else:
        return jsonify({"error": "No state found for the given project ID"}), 404

print("IMPORT_LOG: Finished loading state.py", file=sys.stderr); sys.stderr.flush()