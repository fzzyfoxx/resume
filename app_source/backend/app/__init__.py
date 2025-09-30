from flask import Flask, session, request
from app.routes.filters_routes import filters_bp
from app.routes.queries import queries_bp
from app.routes.state import state_bp
from flask_cors import CORS # Import CORS
from flask_session import Session
import redis
import logging
import sys
import os

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    app.config.from_pyfile('config.py', silent=True) # Load configuration

    # --- Configure Flask-Session ---
    app.config["SESSION_TYPE"] = "redis"           # Using filesystem for server-side storage
    app.config["SESSION_USE_SIGNER"] = True             # Sign the session ID cookie
    app.config["SESSION_PERMANENT"] = True              # Make the session permanent
    app.config['SESSION_REDIS'] = redis.from_url('redis://127.0.0.1:6379')
    Session(app)

    # Add a handler for OPTIONS requests
    @app.before_request
    def handle_options_requests():
        if request.method == 'OPTIONS':
            return '', 200 # Return an empty 200 OK response

    @app.before_request
    def initialize_session():
        # Only initialize if the session is new
        if not session.get('queries'):
            session['queries'] = {'abc': 'xyz'}
        if not session.get('results'):
            session['results'] = {}
        if not session.get('states'):
            session['states'] = {}
            app.logger.debug("Initializing new session data.")

        app.logger.debug(f"Current Session ID: {session.sid}")

    # Register blueprints
    app.register_blueprint(filters_bp, url_prefix='/api/filters')
    app.register_blueprint(queries_bp, url_prefix='/api/queries')
    app.register_blueprint(state_bp, url_prefix='/api/state')


    @app.route('/')
    def index():
        return "Welcome to the GeoDoc API!"

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config.get('DEBUG', True), port=5000)