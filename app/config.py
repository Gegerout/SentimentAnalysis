class Config:
    DEBUG = True
    # Устройство: -1 для CPU, 0 или больше для GPU
    DEVICE = -1
    KAFKA_BROKER_URL = 'kafka:9092'
    KAFKA_TOPIC_DATASET = 'dataset_preparation'
    KAFKA_TOPIC_FINETUNE = 'model_finetune'

    # Имя модели по умолчанию (если чекпоинт не выбран) – либо название из Hugging Face,
    # либо путь к скачанной версии в папке MODEL_CACHE_DIR
    DEFAULT_MODEL_NAME = "blanchefort/rubert-base-cased-sentiment-rusentiment"

    # Папка, куда будут скачиваться (кэшироваться) модели
    MODEL_CACHE_DIR = "./models"

    # Папка, где хранятся дообученные чекпоинты (локальные копии модели после обучения)
    CHECKPOINTS_DIR = "./checkpoints"
