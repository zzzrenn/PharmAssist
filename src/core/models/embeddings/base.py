from abc import ABC, abstractmethod
from typing import List, Union


class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""

    @abstractmethod
    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generates embeddings for the given text(s).

        Args:
            texts: A single string or a list of strings to embed.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            If a single string was input, returns a list containing one embedding.
        """
        pass

    @classmethod
    @abstractmethod
    def from_settings(cls, settings):
        """Factory method to create an instance from the global settings object."""
        pass
