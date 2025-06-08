from typing import List, Union

from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from core.logger_utils import get_logger
from core.models.embeddings.base import EmbeddingModel

logger = get_logger(__name__)


class SparseEmbeddingModel(EmbeddingModel):
    """Sparse embedding model using FastEmbed BM25."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initializes the FastEmbed BM25 embedding model.

        Args:
            model_name: The name of the BM25 model to use (default: "Qdrant/bm25").
        """
        self.model_name = model_name

        try:
            self.model = SparseTextEmbedding(
                model_name=model_name,
            )
            logger.info(
                f"Initialized FastEmbedBM25EmbeddingModel with model '{model_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed BM25 model: {e}")
            raise

    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[SparseEmbedding, List[SparseEmbedding]]:
        """
        Generates sparse BM25 embeddings using FastEmbed.

        Args:
            texts: A single string or list of strings to embed.

        Returns:
            A SparseEmbedding object or a list of SparseEmbedding objects.
        """
        # Ensure texts is a list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        try:
            # Generate sparse embeddings
            embeddings = list(self.model.embed(texts))

            # Return single embedding if single input was provided
            if single_input:
                return embeddings[0]
            else:
                return embeddings

        except Exception as e:
            logger.error(f"Error generating BM25 embeddings: {e}")
            raise

    @classmethod
    def from_settings(cls, settings):
        """Creates an instance from the global settings object."""
        model_name = getattr(settings, "SPARSE_EMBEDDING_MODEL_ID", "Qdrant/bm25")
        return cls(model_name=model_name)
