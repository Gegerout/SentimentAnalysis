from flask import Flask, send_from_directory
from app.config import Config
import os

def create_app():
    # Предположим, что папка website лежит на уровень выше относительно этого файла.
    base_dir = os.path.abspath(os.path.dirname(__file__))
    website_path = os.path.join(base_dir, '..', 'website')

    # Указываем папку для статики и URL, по которому к ней обращаться (в данном случае – корень)
    app = Flask(__name__, static_folder=website_path, static_url_path='')
    app.config.from_object(Config)

    # Регистрируем ваши API blueprints
    from app.routes.inference import inference_bp
    from app.routes.dataset import dataset_bp
    from app.routes.finetune import finetune_bp
    app.register_blueprint(inference_bp, url_prefix='/api')
    app.register_blueprint(dataset_bp, url_prefix='/api')
    app.register_blueprint(finetune_bp, url_prefix='/api')

    # Маршрут для корневого пути, отдающий index.html из папки website
    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    return app
