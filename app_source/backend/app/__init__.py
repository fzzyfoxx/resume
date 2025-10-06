from flask import Flask, session, request
from flask_cors import CORS
from flask_session import Session
import redis
import os
import sys
from werkzeug.local import LocalProxy

# --- This function will help us debug ---
def log_message(message):
    print(f"STARTUP_LOG: {message}", file=sys.stderr)
    sys.stderr.flush()

# A proxy object that will lazily connect to Redis on first use.
def _get_redis_connection():
    log_message("Executing _get_redis_connection")
    redis_host = os.environ.get('REDIS_HOST', '127.0.0.1')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    log_message(f"Attempting to connect to Redis at {redis_host}:{redis_port}")
    # Add timeouts so commands don't hang indefinitely
    conn = redis.Redis(
        host=redis_host,
        port=redis_port,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        socket_keepalive=True,
    )
    log_message("Redis connection object created")
    return conn

lazy_redis = LocalProxy(_get_redis_connection)
flask_session = Session()

def create_app():
    log_message("create_app() called")
    
    log_message("1. Initializing Flask app")
    app = Flask(__name__)
    log_message("1. DONE")

    log_message("2. Initializing CORS")
    CORS(app, supports_credentials=True, origins=["https://perigonscout.pl"])
    log_message("2. DONE")

    log_message("3. Loading config from pyfile")
    # Load config from the package path (not instance path)
    config_path = os.path.join(app.root_path, 'config.py')
    app.config.from_pyfile(config_path, silent=True)
    # Ensure SECRET_KEY is present even if pyfile wasn’t found
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-not-secure")
    log_message("3. DONE")

    log_message("4. Setting Flask-Session config")
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_PERMANENT"] = True
    app.config['SESSION_REDIS'] = lazy_redis
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True
    log_message("4. DONE")
    
    log_message("5. Initializing Flask-Session with init_app")
    flask_session.init_app(app)
    log_message("5. DONE")

    log_message("6. Defining before_request handler")
    # Health endpoint that doesn't touch session
    @app.get('/healthz')
    def healthz():
        return "ok", 200

    # Skip session access for health/readiness paths
    HEALTH_PATHS = {"/", "/healthz", "/_ah/warmup"}
    @app.before_request
    def initialize_session():
        log_message("before_request: initializing session")
        if request.path in HEALTH_PATHS:
            return
        if 'queries' not in session:
            session['queries'] = {'abc': 'xyz'}
        if 'results' not in session:
            session['results'] = {}
        if 'states' not in session:
            session['states'] = {}
        log_message("before_request: DONE")

    log_message("6. DONE")

    log_message("7. Registering blueprints")
    # Import blueprints lazily to keep module import fast
    from app.routes.filters_routes import filters_bp
    from app.routes.queries import queries_bp
    from app.routes.state import state_bp
    app.register_blueprint(filters_bp, url_prefix='/api/filters')
    app.register_blueprint(queries_bp, url_prefix='/api/queries')
    app.register_blueprint(state_bp, url_prefix='/api/state')
    log_message("7. DONE")

    log_message("8. Defining index route")
    @app.route('/')
    def index():
        # health/readiness hits this; before_request guard prevents session I/O
        return "Welcome to the GeoDoc API!"
    log_message("8. DONE")

    log_message("create_app() finished, returning app object")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)


