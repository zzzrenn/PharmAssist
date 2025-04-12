import sys
import sys
import os
# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core import logger_utils
from core.config import settings
from chatbot import Chatbot
from huggingface_hub import login

logger = logger_utils.get_logger(__name__)


if __name__ == "__main__":
    login(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    inference_endpoint = Chatbot(mock=False)

    query = """
    how to treat a pregnant woman with a hypertension?
    """

    response = inference_endpoint.generate(
        query=query, enable_rag=True, sample_for_evaluation=False
    )

    logger.info("=" * 50)
    logger.info(f"Query: {query}")
    logger.info("=" * 50)
    logger.info(f"Answer: {response['answer']}")
    logger.info("=" * 50)