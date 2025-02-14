import uuid
import json
from kafka import KafkaProducer, KafkaConsumer
from app.config import Config


def send_task_and_wait_for_response(task, request_topic, response_topic, timeout=30):
    """
    Отправляет задачу в Kafka и ожидает ответа с указанным correlation_id.

    :param task: Словарь с данными задачи.
    :param request_topic: Топик для отправки задачи.
    :param response_topic: Топик, на который воркер отправит обработанный результат.
    :param timeout: Время ожидания ответа в секундах.
    :return: Ответное сообщение (словарь) или выбрасывает TimeoutError.
    """
    correlation_id = str(uuid.uuid4())
    task['correlation_id'] = correlation_id
    task['reply_to'] = response_topic

    # Создаем KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=Config.KAFKA_BROKER_URL,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    producer.send(request_topic, task)
    producer.flush()

    # Создаем KafkaConsumer для прослушивания reply-топика
    consumer = KafkaConsumer(
        response_topic,
        bootstrap_servers=Config.KAFKA_BROKER_URL,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        consumer_timeout_ms=timeout * 1000  # таймаут в мс
    )

    response = None
    for message in consumer:
        msg = message.value
        if msg.get('correlation_id') == correlation_id:
            response = msg
            break
    consumer.close()

    if response is None:
        raise TimeoutError("Timeout waiting for Kafka response")
    return response
