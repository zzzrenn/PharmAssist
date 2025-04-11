from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # MongoDB configs
    # MONGODB_USERNAME: str
    # MONGODB_PASSWORD: str
    MONGO_DATABASE_HOST: str 
    MONGO_DATABASE_NAME: str = "pharmassist"

    # MQ config
    RABBITMQ_DEFAULT_USERNAME: str = "guest"
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"
    RABBITMQ_HOST: str = "mq"
    RABBITMQ_PORT: int = 5672

    # QdrantDB config
    QDRANT_CLOUD_URL: str = "https://c8820847-221d-42b6-9a77-75afc147c89b.eu-central-1-0.aws.cloud.qdrant.io"
    QDRANT_DATABASE_HOST: str = "qdrant"
    QDRANT_DATABASE_PORT: int = 6333
    USE_QDRANT_CLOUD: bool = True
    QDRANT_APIKEY: str | None = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DEOG7SVnS9BJ6Aar4E7847jPnWLLL108O6jGnaoptDg"

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 384
    EMBEDDING_MODEL_DEVICE: str = "cpu"

    # RAG config
    ENABLE_SELF_QUERY: bool = False
    ENABLE_RERANKING: bool = False

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str

    # CometML config
    COMET_API_KEY: str
    COMET_WORKSPACE: str
    COMET_PROJECT: str = "pharmassist"

    # OPIK config
    OPIK_API_KEY: str

settings = AppSettings()