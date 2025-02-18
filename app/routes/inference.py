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


@inference_bp.route('/predict_text_ensemble', methods=['POST'])
def predict_text_ensemble():
    """
    Эндпоинт для предсказания по одному тексту с использованием ансамблевой модели.
    Ожидается JSON с полем:
      - text: текст для анализа.
    Задача отправляется в Kafka с типом 'predict_text_ensemble'.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Не предоставлен текст для анализа."}), 400

    text = data['text']
    task = {
        'type': 'predict_text_ensemble',
        'text': text,
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

    result = response.get('result')
    if not result:
        return jsonify({"error": "Ответ от воркера не содержит результата."}), 500

    label = result.get("label", "").lower()
    sentiment_letter = SENTIMENT_MAP.get(label, "N")
    return jsonify({
        "result": sentiment_letter,
        "inference_time": elapsed_time
    })


@inference_bp.route('/predict_file_ensemble', methods=['POST'])
def predict_file_ensemble():
    """
    Эндпоинт для предсказания по Excel‑файлу с использованием ансамблевой модели.
    Ожидается multipart/form-data запрос с:
      - file: XLSX‑файл (ожидается наличие столбца 'MessageText')
    Задача отправляется в Kafka с типом 'predict_file_ensemble'.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Не передан файл в поле 'file'."}), 400

    file = request.files['file']
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения Excel‑файла: {str(e)}"}), 400

    if 'MessageText' not in df.columns:
        return jsonify({"error": "В Excel‑файле должен присутствовать столбец 'MessageText'."}), 400

    texts = df['MessageText'].tolist()

    task = {
        'type': 'predict_file_ensemble',
        'texts': texts,
    }

    start_time = time.time()
    try:
        response = send_task_and_wait_for_response(
            task,
            request_topic='inference_request',
            response_topic='inference_response',
            timeout=30
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    elapsed_time = time.time() - start_time

    results = response.get('results')
    if not results:
        return jsonify({"error": "Ответ от воркера не содержит результатов."}), 500

    sentiments = []
    # Предполагаем, что каждый элемент результата – словарь с ключом "label"
    for res in results:
        label = res.get("label", "").lower()
        sentiment_letter = SENTIMENT_MAP.get(label, "N/A")
        sentiments.append(sentiment_letter)

    df['sentiment'] = sentiments

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        meta_df = pd.DataFrame({"inference_time": [elapsed_time]})
        meta_df.to_excel(writer, index=False, sheet_name='Meta')
    output.seek(0)

    return send_file(
        output,
        download_name="result.xlsx",
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@inference_bp.route('/predict_text', methods=['POST'])
def predict_text():
    """
    Эндпоинт для предсказания по одному тексту.
    Ожидается JSON с полями:
      - text: текст для анализа
      - model_name (необязательно): имя модели, которая должна быть использована
    Задача отправляется в Kafka (топик inference_request) с ожиданием ответа.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Не предоставлен текст для анализа."}), 400

    text = data['text']
    model_name = data.get('model_name')  # получаем имя модели

    # Формируем задачу для Kafka
    task = {
        'type': 'predict_text',
        'text': text,
        'model_name': model_name
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
      - model_name (необязательно): имя модели для инференса

    Обработка:
      - Файл считывается с помощью pandas.read_excel.
      - Из таблицы извлекается столбец 'MessageText'.
      - Формируется задача для Kafka (тип 'predict_file') с передачей списка текстов.
      - Ожидается ответ от воркера с результатами.
      - Формируется Excel‑файл с двумя листами:
          - "Predictions" с результатами (новый столбец 'sentiment'),
          - "Meta" с информацией о времени предсказания.
    """
    # Проверяем, что файл передан в поле 'file'
    if 'file' not in request.files:
        return jsonify({"error": "Не передан файл в поле 'file'."}), 400

    file = request.files['file']
    text_column = request.form.get('text_column', "MessageText")
    try:
        # Читаем Excel‑файл в DataFrame
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения Excel‑файла: {str(e)}"}), 400

    # Проверяем наличие столбца с текстами (ожидаем 'MessageText')
    if text_column not in df.columns:
        return jsonify({"error": "В Excel‑файле должен присутствовать столбец 'MessageText'."}), 400

    texts = df[text_column].tolist()

    # Получаем имя модели из формы (если передано)
    model_name = request.form.get('model_name')

    # Формируем задачу для Kafka (тип 'predict_file')
    task = {
        'type': 'predict_file',
        'texts': texts,
        'model_name': model_name
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
        download_name="result.xlsx",
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@inference_bp.route('/predict_file_custom', methods=['POST'])
def predict_file_custom():
    if 'file' not in request.files:
        return jsonify({"error": "Не передан файл в поле 'file'."}), 400

    file = request.files['file']
    text_column = request.form.get('text_column', "MessageText")

    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения Excel‑файла: {str(e)}"}), 400

    if text_column not in df.columns:
        return jsonify({"error": f"В Excel‑файле должен присутствовать столбец '{text_column}'."}), 400

    texts = df[text_column].tolist()
    model_name = request.form.get('model_name')

    start_time = time.time()
    # Флаг, определяющий, какой метод предсказания использовался
    method_used = None
    try:
        # Пытаемся использовать вашу модель
        from metamodels import predict as my_model_predict
        predictions = my_model_predict(texts, verbose=False)
        # Извлекаем сентимент для каждого текста
        sentiments = [pred.get('sentiment', 'error') for pred in predictions]
        method_used = 'my_model'
    except Exception as e:
        # Если произошла ошибка – переходим на обычное предсказание через Kafka
        try:
            task = {
                'type': 'predict_file',
                'texts': texts,
                'model_name': model_name
            }
            response = send_task_and_wait_for_response(
                task,
                request_topic='inference_request',
                response_topic='inference_response',
                timeout=30  # таймаут в секундах
            )
            results = response.get('results')
            if not results:
                return jsonify({"error": "Ответ от воркера не содержит результатов."}), 500
            sentiments = []
            for res in results:
                label = res.get("label", "").lower()
                sentiment_letter = SENTIMENT_MAP.get(label, "N/A")
                sentiments.append(sentiment_letter)
            method_used = 'fallback'
        except Exception as ex:
            return jsonify({"error": f"Ошибка при выполнении предсказаний (fallback): {str(ex)}"}), 500

    elapsed_time = time.time() - start_time

    # Добавляем результаты в DataFrame
    df['sentiment'] = sentiments

    # Формируем Excel‑файл с результатами и метаинформацией
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        meta_df = pd.DataFrame({
            "inference_time": [elapsed_time],
            "method_used": [method_used]
        })
        meta_df.to_excel(writer, index=False, sheet_name='Meta')
    output.seek(0)

    return send_file(
        output,
        download_name="result.xlsx",
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
