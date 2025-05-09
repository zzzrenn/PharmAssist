import argparse

import opik
from config import settings
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, Hallucination, Moderation

from core.logger_utils import get_logger

logger = get_logger(__name__)


def evaluation_task(x: dict) -> dict:
    return {
        "input": x["input"]["query"],
        "context": x["expected_output"]["context"],
        "output": x["expected_output"]["answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG LLM on monitoring dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PharmAssistMonitoringDataset",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

    client = opik.Opik()
    try:
        dataset = client.get_dataset(dataset_name)
    except Exception:
        logger.error(f"Monitoring dataset '{dataset_name}' not found in Opik. Exiting.")
        exit(1)

    experiment_config = {
        "model_id": settings.MODEL_ID,
    }

    scoring_metrics = [Hallucination(), Moderation(), AnswerRelevance()]
    evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=scoring_metrics,
        experiment_config=experiment_config,
        task_threads=1,
    )


if __name__ == "__main__":
    main()
