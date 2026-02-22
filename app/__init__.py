from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    
    from app.routes import api_bp
    app.register_blueprint(api_bp)
    
    return app
