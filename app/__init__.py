from flask import Flask
from app.config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from app.routes.inference import inference_bp
    from app.routes.dataset import dataset_bp
    app.register_blueprint(inference_bp, url_prefix='/api')
    app.register_blueprint(dataset_bp, url_prefix='/api')

    return app
