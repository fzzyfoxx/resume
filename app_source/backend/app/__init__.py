from flask import Flask, session
from app.routes.filters_routes import filters_bp
from app.routes.queries import queries_bp
from flask_cors import CORS # Import CORS
from flask_session import Session
import logging
import sys
import os
print("sys.path:", sys.path)

#app = Flask(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    app.config.from_pyfile('config.py', silent=True) # Load configuration

    # --- Configure Flask-Session ---
    app.config["SESSION_TYPE"] = "filesystem"           # Using filesystem for server-side storage
    app.config["SESSION_USE_SIGNER"] = True             # Sign the session ID cookie
    app.config["SESSION_PERMANENT"] = True              # Make the session permanent
    app.config["SESSION_FILE_DIR"] = "/tmp/flask_session" # Specify where to store files
    Session(app)

    # Check if the session directory exists and is writable
    session_dir = app.config["SESSION_FILE_DIR"]
    if not os.path.exists(session_dir):
        print(f"Session directory '{session_dir}' does not exist. Creating it...")
        os.makedirs(session_dir, exist_ok=True)
    if not os.access(session_dir, os.W_OK):
        raise PermissionError(f"Write permission is denied for the session directory: {session_dir}")
    else:
        print(f"Session directory '{session_dir}' is writable.")


    logging.basicConfig(level=logging.DEBUG)

    @app.before_request
    def initialize_session():
        if 'queries' not in session:
            session['queries'] = {'abc': 'xyz'}  # Initialize with some default data
        if 'results' not in session:
            session['results'] = {}
        #app.logger.debug(f"Session data: {session}")

    # Register blueprints
    app.register_blueprint(filters_bp, url_prefix='/api/filters')
    app.register_blueprint(queries_bp, url_prefix='/api/queries')


    @app.route('/')
    def index():
        return "Welcome to the GeoDoc API!"

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config.get('DEBUG', True), port=5000)