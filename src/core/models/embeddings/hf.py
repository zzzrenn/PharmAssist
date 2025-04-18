from typing import List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer

from core.logger_utils import get_logger

# Adjust import path for the new location
from core.models.embeddings.base import EmbeddingModel

logger = get_logger(__name__)


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Embedding model using Hugging Face's sentence-transformers."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initializes the HuggingFace embedding model.

        Args:
            model_name: The name of the sentence-transformer model to use.
            device: The device to run the model on ('cpu', 'cuda', etc.). Auto-detected if None.
        """
        self.model_name = model_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            logger.warning(
                "CUDA device requested but not available. Falling back to CPU."
            )
        else:
            logger.info("Using CUDA device for embeddings.")
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Initialized HuggingFaceEmbeddingModel with model '{model_name}'")

    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Generates embeddings using the loaded sentence-transformer model."""

        return self.model.encode(texts).tolist()

    @classmethod
    def from_settings(cls, settings):
        """Creates an instance from the global settings object."""
        return cls(
            model_name=settings.EMBEDDING_MODEL_ID,
            device=settings.EMBEDDING_MODEL_DEVICE,
        )
