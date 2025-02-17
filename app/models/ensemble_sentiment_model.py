import re
import numpy as np
import torch
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import joblib
from app.config import Config  # предполагается, что в конфиге задан MODEL_CACHE_DIR

# Если стоп-слова ещё не скачаны
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")


class EnsembleSentimentModel:
    def __init__(self, transformer_model_name=None, device=None):
        """
        Инициализация ансамблевой модели.
        :param transformer_model_name: Имя или путь к трансформер-модели.
        :param device: Устройство для выполнения (0 для GPU, -1 для CPU).
        """
        self.transformer_model_name = transformer_model_name or "blanchefort/rubert-base-cased-sentiment-rusentiment"
        self.device = device if device is not None else (0 if torch.cuda.is_available() else -1)

        # Загружаем трансформер-пайплайн для анализа тональности
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.transformer_model_name,
            tokenizer=self.transformer_model_name,
            device=self.device,
            framework="pt"
        )

        # Инициализируем классическую модель (TF-IDF + LogisticRegression).
        # При обучении модель сохраняется в виде pickle-файлов.
        self.classic_pipeline = None

        # Мета-модель (будет подгружена из кеша)
        self.meta_model = None

    @staticmethod
    def clean_html_tags(text):
        """Удаляет HTML-теги из текста."""
        return re.sub(r'<.*?>', '', text).strip()

    @staticmethod
    def custom_preprocessor(text):
        """Приводит текст к нижнему регистру, удаляет цифры, спецсимволы и выполняет стемминг."""
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        return " ".join(tokens)

    def get_transformer_probs(self, text):
        """
        Получает вектор вероятностей от трансформер-модели.
        Возвращает список из 3 чисел в порядке:
          [вероятность для NEGATIVE (2), вероятность для POSITIVE (1), вероятность для NEUTRAL (0)]
        """
        cleaned = self.clean_html_tags(text)
        scores = self.sentiment_analyzer(cleaned, return_all_scores=True)
        score_dict = {item["label"]: item["score"] for item in scores[0]}
        return [
            score_dict.get("NEGATIVE", 0),  # метка 2
            score_dict.get("POSITIVE", 0),  # метка 1
            score_dict.get("NEUTRAL", 0)  # метка 0
        ]

    def get_classic_probs(self, text):
        """
        Получает вектор вероятностей от классической модели (TF-IDF + LogisticRegression).
        Формирует вектор в порядке: [вероятность для 2, вероятность для 1, вероятность для 0].
        """
        probs = self.classic_pipeline.predict_proba([text])[0]
        prob_dict = {cls: prob for cls, prob in zip(self.classic_pipeline.classes_, probs)}
        return [
            prob_dict.get(2, 0),  # вероятность для метки 2 (негативное)
            prob_dict.get(1, 0),  # вероятность для метки 1 (позитивное)
            prob_dict.get(0, 0)  # вероятность для метки 0 (нейтральное)
        ]

    def get_transformer_pred(self, text):
        """
        Получает предсказание от трансформер-модели.
        Маппит строковую метку в число:
          'NEGATIVE' -> 2, 'POSITIVE' -> 1, 'NEUTRAL' -> 0.
        """
        cleaned = self.clean_html_tags(text)
        result = self.sentiment_analyzer(cleaned, truncation=True, max_length=512)
        label = result[0]['label']
        mapping = {"NEGATIVE": 2, "POSITIVE": 1, "NEUTRAL": 0}
        return mapping.get(label, -1)

    def get_classic_pred(self, text):
        """
        Получает предсказание от классической модели (TF-IDF + LogisticRegression).
        """
        return self.classic_pipeline.predict([text])[0]

    def get_meta_features(self, text):
        """
        Формирует вектор признаков для мета-модели размерностью 2.
        Используются предсказания базовых моделей:
          - первое значение: предсказание трансформер-модели,
          - второе значение: предсказание классической модели.
        """
        transformer_pred = self.get_transformer_pred(text)
        classic_pred = self.get_classic_pred(text)
        print(transformer_pred, classic_pred)
        return np.array([[transformer_pred, classic_pred]])

    def predict(self, text):
        """
        Выполняет предсказание для одного текста с использованием мета-модели.
        Возвращает итоговую метку в виде буквы ("B", "G", "N").
        """
        meta_feat = self.get_meta_features(text)
        pred_numeric = self.meta_model.predict(meta_feat)[0]
        mapping_back = {2: "B", 1: "G", 0: "N"}
        return mapping_back.get(pred_numeric, pred_numeric)

    def predict_batch(self, texts):
        """
        Выполняет предсказание для списка текстов.
        Возвращает список итоговых меток (буквы: "B", "G", "N").
        """
        meta_features = np.concatenate([self.get_meta_features(text) for text in texts], axis=0)
        preds_numeric = self.meta_model.predict(meta_features)
        mapping_back = {2: "B", 1: "G", 0: "N"}
        return [mapping_back.get(pred, pred) for pred in preds_numeric]

    def load_cached_models(self):
        """
        Загружает предобученные модели для классической части и мета-модели
        из указанной директории (MODEL_CACHE_DIR).
        Ожидается, что классическая модель сохранена в 'logistic.pkl',
        а мета-модель – в 'meta.pkl'.
        """
        classic_path = os.path.join(Config.MODEL_CACHE_DIR, "logistic.pkl")
        meta_path = os.path.join(Config.MODEL_CACHE_DIR, "meta.pkl")

        self.classic_pipeline = joblib.load(classic_path)
        self.meta_model = joblib.load(meta_path)
