from app import create_app, flask_session

# Create the core Flask application object
app = create_app()

# Now, initialize the session extension with the app object.
# This happens after create_app is finished and just before Gunicorn takes over.
flask_session.init_app(app)