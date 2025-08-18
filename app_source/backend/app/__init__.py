from flask import Flask
from app.routes.filters_routes import filters_bp
from flask_cors import CORS # Import CORS

import sys
print("sys.path:", sys.path)

#app = Flask(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config.from_pyfile('config.py', silent=True) # Load configuration

    # Register blueprints
    app.register_blueprint(filters_bp, url_prefix='/api/filters')

    @app.route('/')
    def index():
        return "Welcome to the GeoDoc API!"

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config.get('DEBUG', True), port=5000)