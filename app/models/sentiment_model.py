import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from app.config import Config

# Настройка потоков для оптимизации на CPU
torch.set_num_threads(8)
torch.set_num_interop_threads(8)


class SentimentModel:
    def __init__(self, model_path=None):
        """
        Если model_path не указан, используется модель по умолчанию.
        model_path может быть либо именем модели для скачивания,
        либо локальным путём к модели.
        Если базовое имя model_path начинается с "models--",
        то этот префикс удаляется, а все последующие "--" заменяются на "/"
        для формирования корректного идентификатора модели.
        """
        self.model_path = model_path or Config.DEFAULT_MODEL_NAME
        model_id = self.model_path

        # Загружаем токенизатор и модель с указанием кэш-директории
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=Config.MODEL_CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=Config.MODEL_CACHE_DIR)

        # Создаем пайплайн – он использует уже загруженные модель и токенизатор
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=Config.DEVICE,
            framework="pt"
        )

    def predict(self, text, **pipeline_kwargs):
        """
        Выполняет предсказание для одного текста.
        Дополнительные параметры можно передать через pipeline_kwargs.
        """
        results = self.sentiment_analyzer(text, **pipeline_kwargs)
        return results[0] if isinstance(results, list) else results

    def predict_batch(self, texts, **pipeline_kwargs):
        """
        Выполняет предсказание для списка текстов.
        """
        return self.sentiment_analyzer(texts, **pipeline_kwargs)
