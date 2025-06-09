from typing import Dict, Type

from inference_pipeline.chatbots.chatbot_base import ChatbotBase
from inference_pipeline.chatbots.openai import ChatbotOpenAI
from inference_pipeline.chatbots.qwen import ChatbotQwen
from inference_pipeline.config import settings

# Registry of available embedding models
_CHATBOT_REGISTRY: Dict[str, Type[ChatbotBase]] = {
    "openai": ChatbotOpenAI,
    "qwen": ChatbotQwen,
    # Add other providers here
}


class ChatbotFactory:
    """
    Factory to create chatbot instances using global settings.
    """

    def __init__(self, settings=settings):
        """Initializes the factory, optionally using a specific settings instance."""
        self.settings = settings

    def create_chatbot(self, provider: str, mock: bool = False) -> ChatbotBase:
        """
        Creates a chatbot instance based on the provider name.

        Args:
            provider: The name of the chatbot provider (e.g., 'openai', 'qwen').

        Returns:
            An instance of the requested EmbeddingModel or None (only for sparse embeddings).
        """
        provider_lower = provider.lower()

        if provider_lower not in _CHATBOT_REGISTRY:
            raise ValueError(
                f"Unknown chatbot provider: '{provider}'. "
                f"Available: {list(_CHATBOT_REGISTRY.keys())}"
            )
        chatbot_cls = _CHATBOT_REGISTRY[provider_lower]
        try:
            # Use the classmethod with the potentially overridden settings
            return chatbot_cls(mock=mock)
        except AttributeError as e:
            raise AttributeError(
                f"Missing required setting for provider '{provider}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error creating chatbot for provider '{provider}': {e}"
            ) from e
