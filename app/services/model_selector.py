import os
from app.config import Config
from app.models.sentiment_model import SentimentModel


def list_available_checkpoints():
    """
    Возвращает список файлов-чекпоинтов (например, с расширением .ckpt),
    найденных в папке, указанной в Config.CHECKPOINTS_DIR.
    """
    checkpoints = []
    if not os.path.exists(Config.CHECKPOINTS_DIR):
        return checkpoints  # Папки нет — пустой список
    for filename in os.listdir(Config.CHECKPOINTS_DIR):
        if filename.endswith('.ckpt'):
            checkpoints.append(filename)
    return checkpoints


def select_model(checkpoint_name=None):
    """
    Если checkpoint_name передан и присутствует в CHECKPOINTS_DIR,
    то модель загружается из локального чекпоинта.
    Иначе используется модель по умолчанию.
    """
    if checkpoint_name:
        available = list_available_checkpoints()
        if checkpoint_name not in available:
            raise Exception(f"Запрошенный чекпоинт '{checkpoint_name}' недоступен. Доступны: {available}")
        # Формируем полный путь к файлу-чекпоинту
        model_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
    else:
        model_path = Config.DEFAULT_MODEL_NAME
    return SentimentModel(model_path=model_path)
