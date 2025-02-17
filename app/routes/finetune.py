import json
import time

from flask import Blueprint, jsonify, request, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import Config
from app.services.model_selector import list_available_models
import pandas as pd

finetune_bp = Blueprint('finetune', __name__)

@finetune_bp.route('/models', methods=['GET'])
def get_models():
    models = list_available_models()
    return jsonify(models)


@finetune_bp.route('/download_model', methods=['GET'])
def download_model():
    """
    Эндпоинт принимает GET-запрос с параметром "model_name",
    запускает скачивание токенизатора и модели из Hugging Face
    с сохранением в Config.MODEL_CACHE_DIR и возвращает обновления
    прогресса через SSE.
    """
    model_name = request.args.get("model_name")
    if not model_name:
        return jsonify({"error": "Параметр model_name обязателен"}), 400

    def generate():
        try:
            # Начало скачивания
            yield f"data: {json.dumps({'progress': 0, 'message': f'Начало скачивания модели {model_name}'})}\n\n"

            # Скачиваем токенизатор
            yield f"data: {json.dumps({'progress': 10, 'message': 'Скачивание токенизатора...'})}\n\n"
            AutoTokenizer.from_pretrained(model_name, cache_dir=Config.MODEL_CACHE_DIR)
            yield f"data: {json.dumps({'progress': 50, 'message': 'Токенизатор скачан.'})}\n\n"

            # Скачиваем модель
            yield f"data: {json.dumps({'progress': 60, 'message': 'Скачивание модели...'})}\n\n"
            AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=Config.MODEL_CACHE_DIR)
            yield f"data: {json.dumps({'progress': 100, 'message': 'Модель скачана.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'progress': -1, 'message': f'Ошибка при скачивании: {str(e)}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@finetune_bp.route('/finetune', methods=['GET'])
def finetune():
    # Получаем параметры из query-строки
    epochs = request.args.get("epochs")
    learning_rate = request.args.get("learning_rate")
    batch_size = request.args.get("batch_size")

    # Проверяем, что все параметры переданы
    if not epochs or not learning_rate or not batch_size:
        return jsonify({"error": "Параметры epochs, learning_rate и batch_size обязательны"}), 400

    # Приводим типы параметров
    try:
        epochs = int(epochs)
        learning_rate = float(learning_rate)
        batch_size = int(batch_size)
    except ValueError:
        return jsonify({"error": "Неверный формат параметров"}), 400

    # Определяем задержку для каждой эпохи.
    # Для симуляции обучения используем learning_rate * 10 секунд.
    # (Поскольку типичные значения learning_rate достаточно малы, домножаем на константу)
    per_epoch_delay = batch_size * learning_rate * 1000

    def generate():
        # Отправляем сообщение о старте обучения
        yield f"data: {json.dumps({'progress': 0, 'message': 'Начало обучения'})}\n\n"

        # Симуляция обучения по эпохам
        for epoch in range(1, epochs + 1):
            time.sleep(per_epoch_delay)  # имитация длительности эпохи
            progress = int((epoch / epochs) * 100)
            yield f"data: {json.dumps({'progress': progress, 'message': f'Эпоха {epoch}/{epochs} завершена. Batch size: {batch_size}'})}\n\n"

        # Отправляем сообщение о завершении обучения
        yield f"data: {json.dumps({'progress': 100, 'message': 'Обучение завершено.'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
