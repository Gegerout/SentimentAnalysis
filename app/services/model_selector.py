import os
from app.config import Config
from app.models.sentiment_model import SentimentModel


def list_available_models():
    """
    Возвращает список доступных моделей, представленных папками,
    найденных в папке, указанной в Config.MODEL_CACHE_DIR.
    Скрытые папки (начинающиеся с точки) исключаются.
    Если имя папки начинается с "models--", то возвращается нормализованное имя:
      - префикс "models--" удаляется,
      - оставшиеся вхождения "--" заменяются на "/".
    """
    models = []
    if not os.path.exists(Config.MODEL_CACHE_DIR):
        return models
    for item in os.listdir(Config.MODEL_CACHE_DIR):
        if item.startswith('.'):
            continue
        item_path = os.path.join(Config.MODEL_CACHE_DIR, item)
        if os.path.isdir(item_path):
            if item.startswith("models--"):
                normalized_name = item[len("models--"):].replace("--", "/")
                models.append(normalized_name)
            else:
                models.append(item)
    return models


def select_model(model_name=None):
    """
    Если model_name передано и присутствует в MODEL_CACHE_DIR,
    то модель загружается из соответствующей папки.
    Иначе используется модель по умолчанию.
    """
    if model_name:
        available = list_available_models()
        if model_name not in available:
            raise Exception(f"Запрошенная модель '{model_name}' недоступна. Доступны: {available}")
        return SentimentModel(model_path=model_name)
    else:
        return SentimentModel(model_path=Config.DEFAULT_MODEL_NAME)

