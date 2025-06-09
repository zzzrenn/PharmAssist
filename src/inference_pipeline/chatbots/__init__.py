from inference_pipeline.chatbots.chatbot_factory import ChatbotFactory
from inference_pipeline.config import settings

# Global instance of the factory using the default global settings
chatbot_factory = ChatbotFactory()
chatbot = chatbot_factory.create_chatbot(provider=settings.CHATBOT_PROVIDER)

__all__ = ["chatbot", "chatbot_factory"]
