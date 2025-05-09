import argparse

from chatbot import Chatbot
from config import settings
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    ContextPrecision,
    ContextRecall,
    Hallucination,
)

from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts

# settings.patch_localhost()

logger = get_logger(__name__)
# logger.warning(
#     "Patched settings to work with 'localhost' URLs. \
#     Remove the 'settings.patch_localhost()' call from above when deploying or running inside Docker."
# )


def evaluation_task(x: dict) -> dict:
    inference_pipeline = Chatbot(mock=False)
    result = inference_pipeline.generate(
        query=x["instruction"],
        enable_rag=True,
    )
    answer = result["answer"]
    context = result["context"]

    return {
        "input": x["instruction"],
        "output": answer,
        "context": context,
        "expected_output": x["content"],
        "reference": x["content"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG on test dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PharmAssistTestDataset",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name

    logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

    dataset = create_dataset_from_artifacts(
        dataset_name=dataset_name,
        artifact_names=[
            "NICE_Guideline-instruct-dataset",
        ],
    )
    if dataset is None:
        logger.error("Dataset can't be created. Exiting.")
        exit(1)

    experiment_config = {
        "model_id": settings.MODEL_ID,
        "embedding_model_id": settings.EMBEDDING_MODEL_ID,
    }
    scoring_metrics = [
        Hallucination(),
        ContextRecall(),
        ContextPrecision(),
    ]
    evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=scoring_metrics,
        experiment_config=experiment_config,
        task_threads=1,
    )


if __name__ == "__main__":
    main()
