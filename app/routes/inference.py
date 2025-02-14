import time
from flask import Blueprint, request, send_file, jsonify
import pandas as pd
from io import BytesIO
from app.services.kafka_producer import send_task_and_wait_for_response

# Словарь для преобразования меток модели в требуемые символы
SENTIMENT_MAP = {
    "negative": "B",  # negative -> B
    "positive": "G",  # positive -> G
    "neutral": "N"    # neutral  -> N
}

inference_bp = Blueprint('inference', __name__)


@inference_bp.route('/predict_text', methods=['POST'])
def predict_text():
    """
    Эндпоинт для предсказания по одному тексту.
    Ожидается JSON с полями:
      - text: текст для анализа
      - checkpoint (необязательно): имя чекпоинта (например, "checkpoint_v2.ckpt")

    Задача отправляется в Kafka (топик inference_request) с ожиданием ответа
    (в топике inference_response). Ответ возвращается клиенту вместе с временем предсказания.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Не предоставлен текст для анализа."}), 400

    text = data['text']
    checkpoint = data.get('checkpoint')

    # Формируем задачу для Kafka
    task = {
        'type': 'predict_text',
        'text': text,
        'checkpoint': checkpoint
    }

    start_time = time.time()
    try:
        response = send_task_and_wait_for_response(
            task,
            request_topic='inference_request',
            response_topic='inference_response',
            timeout=30  # таймаут в секундах
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    elapsed_time = time.time() - start_time

    # Ожидаем, что воркер вернул результат в поле "result"
    result = response.get('result')
    if not result:
        return jsonify({"error": "Ответ от воркера не содержит результата."}), 500

    label = result.get("label", "").lower()
    sentiment_letter = SENTIMENT_MAP.get(label, "N/A")
    return jsonify({
        "result": sentiment_letter,
        "inference_time": elapsed_time  # время предсказания в секундах
    })


@inference_bp.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Эндпоинт для предсказания по Excel‑файлу.

    Ожидается multipart/form-data запрос с:
      - file: XLSX‑файл (ожидается наличие столбца 'MessageText')

    Обработка:
      - Файл считывается с помощью pandas.read_excel.
      - Из таблицы извлекается столбец 'MessageText' (список текстов).
      - Формируется задача для Kafka (тип 'predict_file') с передачей списка текстов.
      - Ожидается ответ от воркера с результатами (список, где для каждого текста указан результат).
      - На основе результатов формируется Excel‑файл, в который добавляются два листа:
          - "Predictions" с результатами (новый столбец 'sentiment'),
          - "Meta" с информацией о времени предсказания.
    """
    # Проверяем, что файл передан в поле 'file'
    if 'file' not in request.files:
        return jsonify({"error": "Не передан файл в поле 'file'."}), 400

    file = request.files['file']
    try:
        # Читаем Excel‑файл в DataFrame
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения Excel‑файла: {str(e)}"}), 400

    # Проверяем наличие столбца с текстами (ожидаем 'MessageText')
    if 'MessageText' not in df.columns:
        return jsonify({"error": "В Excel‑файле должен присутствовать столбец 'MessageText'."}), 400

    texts = df['MessageText'].tolist()

    # Формируем задачу для Kafka (тип 'predict_file')
    task = {
        'type': 'predict_file',
        'texts': texts
    }

    start_time = time.time()
    try:
        response = send_task_and_wait_for_response(
            task,
            request_topic='inference_request',
            response_topic='inference_response',
            timeout=30  # таймаут в секундах
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    elapsed_time = time.time() - start_time

    # Ожидаем, что воркер вернет результаты в поле "results" (список ответов для каждого текста)
    results = response.get('results')
    if not results:
        return jsonify({"error": "Ответ от воркера не содержит результатов."}), 500

    sentiments = []
    for res in results:
        label = res.get("label", "").lower()
        sentiment_letter = SENTIMENT_MAP.get(label, "N/A")
        sentiments.append(sentiment_letter)

    # Добавляем новый столбец с результатами в DataFrame
    df['sentiment'] = sentiments

    # Записываем DataFrame в Excel‑файл в памяти с дополнительным листом с информацией о времени предсказания
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        # Лист с метаинформацией
        meta_df = pd.DataFrame({"inference_time": [elapsed_time]})
        meta_df.to_excel(writer, index=False, sheet_name='Meta')
    output.seek(0)

    # Возвращаем полученный файл как вложение
    return send_file(
        output,
        download_name="result.xlsx",  # для Flask 2.x используем download_name
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
