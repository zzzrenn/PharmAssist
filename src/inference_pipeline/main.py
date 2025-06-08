import argparse
import os
import sys

# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from chatbot import Chatbot
from huggingface_hub import login

from core import logger_utils
from core.config import settings

logger = logger_utils.get_logger(__name__)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Inference pipeline for the chatbot.")
    parser.add_argument(
        "--query",
        type=str,
        default="first line of treatment for pregnant woman with hypertension?",
        help="The query to send to the chatbot.",
    )
    args = parser.parse_args()

    login(token=settings.HUGGINGFACE_ACCESS_TOKEN)
    inference_endpoint = Chatbot(mock=False)

    query = args.query

    response = inference_endpoint.generate(
        query=query, enable_rag=True, sample_for_evaluation=True
    )

    logger.info("=" * 50)
    logger.info(f"Query: {query}")
    logger.info("=" * 50)
    logger.info(f"Answer: {response['answer']}")
    logger.info("=" * 50)
