import json
from kafka import KafkaConsumer, KafkaProducer
from app.config import Config
from app.models.ensemble_sentiment_model import EnsembleSentimentModel
from app.services.model_selector import select_model

model = select_model()

def start_worker():
    """
    Запускает воркера, который слушает топики 'dataset_preparation' и 'inference_request'.
    В зависимости от типа задачи (поле 'type') выполняется обработка:
      - 'prepare_dataset': обрабатывает задачу подготовки датасета,
      - 'predict_text': выполняет инференс для одиночного текста,
      - 'predict_file': выполняет инференс для списка текстов (из файла).
    Результат отправляется в reply-топик (по умолчанию 'dataset_response' для датасета,
    'inference_response' для инференса) с тем же 'correlation_id'.
    """
    consumer = KafkaConsumer(
        'dataset_preparation', 'inference_request',
        bootstrap_servers=Config.KAFKA_BROKER_URL,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest'
    )
    producer = KafkaProducer(
        bootstrap_servers=Config.KAFKA_BROKER_URL,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    for message in consumer:
        task = message.value
        correlation_id = task.get('correlation_id')
        task_type = task.get('type')

        if task_type == 'prepare_dataset':
            processed_data = task.get('data')
            response = {
                'correlation_id': correlation_id,
                'processed_data': processed_data
            }
            reply_to = task.get('reply_to', 'dataset_response')

        elif task_type == 'predict_text':
            # Инференс для одиночного текста
            text = task.get('text')
            model_name = task.get('model_name')  # получаем имя модели из задачи
            try:
                model = select_model(model_name)
                result = model.predict(text, truncation=True, max_length=512)
                response = {'correlation_id': correlation_id, 'result': result}
            except Exception as e:
                response = {'correlation_id': correlation_id, 'error': str(e)}
            reply_to = task.get('reply_to', 'inference_response')

        elif task_type == 'predict_file':
            # Инференс для файла: список текстов
            texts = task.get('texts')
            model_name = task.get('model_name')  # получаем имя модели из задачи
            try:
                model = select_model(model_name)
                results = model.predict_batch(texts, batch_size=16, truncation=True, max_length=512)
                response = {'correlation_id': correlation_id, 'results': results}
            except Exception as e:
                response = {'correlation_id': correlation_id, 'error': str(e)}
            reply_to = task.get('reply_to', 'inference_response')
        elif task_type == 'predict_text_ensemble':
            # Обработка одиночного предсказания ансамблевой модели
            text = task.get('text')
            try:
                model = EnsembleSentimentModel()
                model.load_cached_models()  # подгружаем логистическую и мета-модель из кеша
                result = model.predict(text)
                # Оборачиваем результат в словарь с ключом "label"
                response = {'correlation_id': correlation_id, 'result': {'label': result}}
            except Exception as e:
                response = {'correlation_id': correlation_id, 'error': str(e)}
            reply_to = task.get('reply_to', 'inference_response')

        elif task_type == 'predict_file_ensemble':
            # Обработка пакетного предсказания ансамблевой модели
            texts = task.get('texts')
            try:
                model = EnsembleSentimentModel()
                model.load_cached_models()
                results = model.predict_batch(texts)
                # Формируем список словарей для единообразия
                response = {'correlation_id': correlation_id, 'results': [{'label': r} for r in results]}
            except Exception as e:
                response = {'correlation_id': correlation_id, 'error': str(e)}
            reply_to = task.get('reply_to', 'inference_response')
        else:
            response = {'correlation_id': correlation_id, 'error': 'Unknown task type'}
            reply_to = task.get('reply_to', 'unknown_response')

        # Отправляем ответ в соответствующий reply-топик
        producer.send(reply_to, response)
        producer.flush()
