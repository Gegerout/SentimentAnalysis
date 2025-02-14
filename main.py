import multiprocessing
from app import create_app
from app.worker import start_worker


def run_flask():
    app = create_app()
    # Указываем host="0.0.0.0", чтобы сервер слушал все интерфейсы, и был доступен извне контейнера.
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)


if __name__ == '__main__':
    # Запускаем воркер в отдельном процессе
    worker_process = multiprocessing.Process(target=start_worker)
    worker_process.start()

    # Запускаем Flask-сервер
    run_flask()

    # Когда Flask-сервер завершится, завершаем воркер
    worker_process.join()
