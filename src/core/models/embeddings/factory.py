from typing import Dict, Type

from core.config import settings
from core.models.embeddings.base import EmbeddingModel
from core.models.embeddings.fastembed import SparseEmbeddingModel
from core.models.embeddings.hf import HuggingFaceEmbeddingModel
from core.models.embeddings.openai import OpenAIEmbeddingModel

# Registry of available embedding models
_EMBEDDING_MODEL_REGISTRY: Dict[str, Type[EmbeddingModel]] = {
    "huggingface": HuggingFaceEmbeddingModel,
    "openai": OpenAIEmbeddingModel,
    "fastembed": SparseEmbeddingModel,
    # Add other providers here
}


class EmbeddingModelFactory:
    """
    Factory to create embedding model instances using global settings.
    """

    def __init__(self, settings=settings):
        """Initializes the factory, optionally using a specific settings instance."""
        self.settings = settings

    def create_embedding_model(self, sparse: bool = False) -> EmbeddingModel | None:
        """
        Creates an embedding model instance based on the provider name.

        Args:
            provider: The name of the embedding provider (e.g., 'huggingface', 'openai').
            sparse: Whether to create a sparse embedding model.

        Returns:
            An instance of the requested EmbeddingModel or None (only for sparse embeddings).
        """
        provider = (
            self.settings.EMBEDDING_MODEL_PROVIDER
            if not sparse
            else self.settings.SPARSE_EMBEDDING_MODEL_PROVIDER
        )
        provider_lower = provider.lower()

        # Optional sparse embedding model
        if sparse and provider_lower == "none":
            return None

        if provider_lower not in _EMBEDDING_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Available: {list(_EMBEDDING_MODEL_REGISTRY.keys())}"
            )
        model_cls = _EMBEDDING_MODEL_REGISTRY[provider_lower]
        try:
            # Use the classmethod with the potentially overridden settings
            return model_cls.from_settings(self.settings)
        except AttributeError as e:
            raise AttributeError(
                f"Missing required setting for provider '{provider}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error creating embedding model for provider '{provider}': {e}"
            ) from e


# Global instance of the factory using the default global settings
embedding_model_factory = EmbeddingModelFactory()
