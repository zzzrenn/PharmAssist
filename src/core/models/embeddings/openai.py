from typing import List, Union

from langchain_openai import OpenAIEmbeddings

from core.logger_utils import get_logger
from core.models.embeddings.base import EmbeddingModel

logger = get_logger(__name__)


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model using OpenAI API (potentially via Langchain)."""

    def __init__(self, model_name: str):
        """
        Initializes the OpenAI embedding model.

        Args:
            model_name: The name of the OpenAI embedding model to use.
        """
        self.model_name = model_name
        self.model = OpenAIEmbeddings(model=model_name)
        logger.info(f"Initialized OpenAIEmbeddingModel with model '{model_name}'")

    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Generates embeddings using the OpenAI API."""
        return self.model.embed_documents(texts)

    @classmethod
    def from_settings(cls, settings):
        """Creates an instance from the global settings object."""
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in the configuration.")
        return cls(
            model_name=settings.EMBEDDING_MODEL_ID,
        )
