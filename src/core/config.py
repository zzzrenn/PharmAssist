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
    QDRANT_CLOUD_URL: str
    QDRANT_DATABASE_HOST: str = "qdrant"
    QDRANT_DATABASE_PORT: int = 6333
    USE_QDRANT_CLOUD: bool = True
    QDRANT_APIKEY: str | None

    # Embeddings config
    EMBEDDING_MODEL_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 384
    EMBEDDING_MODEL_DEVICE: str = "cuda"

    # Sparse Embeddings config
    SPARSE_EMBEDDING_MODEL_PROVIDER: str = "fastembed"
    SPARSE_EMBEDDING_MODEL_ID: str = "Qdrant/bm25"

    # RAG config
    ENABLE_SELF_QUERY: bool = True
    ENABLE_RERANKING: bool = True

    # OpenAI config
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str

    # CometML config
    COMET_API_KEY: str
    COMET_WORKSPACE: str
    COMET_PROJECT: str = "pharmassist"

    # OPIK config
    OPIK_API_KEY: str

    # HuggingFace config
    HUGGINGFACE_ACCESS_TOKEN: str

    # AWS Authentication
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None


settings = AppSettings()
