from flask import Blueprint, request, jsonify, send_file
from app.services.kafka_producer import send_task_and_wait_for_response
import pandas as pd
import re
from bs4 import BeautifulSoup
from io import BytesIO

dataset_bp = Blueprint('dataset', __name__)


@dataset_bp.route('/prepare_dataset', methods=['POST'])
def prepare_dataset():
    """
    Ожидаемые входные данные:
       - Формат запроса: multipart/form-data.
       - Ключ 'file' с Excel-файлом.
       - Форм-поля:
             text_column - имя столбца с текстом для анализа,
             sentiment_column - имя столбца с метками тональности.

    Обработка:
       - Excel-файл считывается.
       - Оставляются только два указанных столбца, переименовываются в "TextAnalyze" и "Sentiment".
       - Колонка TextAnalyze очищается от HTML-тегов и лишних пробелов.
       - Затем сформированная задача отправляется через Kafka, и endpoint ожидает обработанный результат.
       - В ответ возвращается обработанный датасет в виде Excel-файла.
    """
    # Проверяем, что файл передан
    if 'file' not in request.files:
        return jsonify({'error': "Файл не передан. Ожидается Excel-файл в поле 'file'."}), 400

    file = request.files['file']
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({'error': f'Ошибка чтения Excel-файла: {str(e)}'}), 400

    # Получаем имена столбцов из form-полей
    text_column = request.form.get('text_column')
    sentiment_column = request.form.get('sentiment_column')

    if not text_column or not sentiment_column:
        return jsonify({'error': 'Нужно указать text_column и sentiment_column в форме'}), 400

    if text_column not in df.columns or sentiment_column not in df.columns:
        return jsonify({
            'error': f"Указанные столбцы отсутствуют в файле. Доступные: {list(df.columns)}"
        }), 400

    # Оставляем только нужные столбцы и переименовываем их
    df = df[[text_column, sentiment_column]].rename(
        columns={text_column: "TextAnalyze", sentiment_column: "Sentiment"}
    )

    # Функция для очистки текста от HTML-тегов и лишних пробелов
    def clean_text(text):
        if not isinstance(text, str):
            return text
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text(separator=" ", strip=True)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    df["TextAnalyze"] = df["TextAnalyze"].apply(clean_text)

    # Преобразуем DataFrame в список словарей
    cleaned_data = df.to_dict(orient="records")

    # Формируем задачу для обработки
    task = {'type': 'prepare_dataset', 'data': cleaned_data}

    try:
        # Отправляем задачу в Kafka и ждем ответа от воркера
        response = send_task_and_wait_for_response(
            task,
            request_topic='dataset_preparation',
            response_topic='dataset_response',
            timeout=30  # таймаут в секундах
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Предполагаем, что воркер возвращает обработанный датасет в ключе 'processed_data'
    processed_data = response.get('processed_data')
    if processed_data is None:
        return jsonify({'error': 'Нет данных в ответе от воркера'}), 500

    # Преобразуем полученные данные (список словарей) в DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Записываем DataFrame в Excel-файл в памяти
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        processed_df.to_excel(writer, index=False)
    output.seek(0)

    # Возвращаем Excel-файл в качестве ответа
    return send_file(
        output,
        as_attachment=True,
        download_name='processed_dataset.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
