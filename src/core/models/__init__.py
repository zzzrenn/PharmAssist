"""Core model components including embeddings, chat models, etc."""

from .embeddings import embedding_model_factory

__all__ = [
    "embedding_model_factory",
]
