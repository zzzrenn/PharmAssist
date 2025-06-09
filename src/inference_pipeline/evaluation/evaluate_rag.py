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

logger = get_logger(__name__)


class RAGEvaluator:
    def __init__(self):
        self.chatbot = Chatbot(mock=False)

    def evaluation_task(self, x: dict) -> dict:
        result = self.chatbot.generate(
            query=x["query"],
            enable_rag=True,
        )
        answer = result["answer"]
        context = result["context"]

        return {
            "input": x["query"],
            "output": answer,
            "context": context,
            "expected_output": x["gt_answer"],
            "reference": x["gt_answer"],
        }

    def evaluate_dataset(self, dataset_name: str) -> None:
        logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

        dataset = create_dataset_from_artifacts(
            dataset_name=dataset_name,
            artifact_names=[
                "cleaned_nice-evaluation-dataset",
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
            Hallucination(model="gpt-4o-mini"),
            ContextRecall(model="gpt-4o-mini"),
            ContextPrecision(model="gpt-4o-mini"),
        ]
        evaluate(
            dataset=dataset,
            task=self.evaluation_task,
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            task_threads=1,
        )


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

    evaluator = RAGEvaluator()
    evaluator.evaluate_dataset(dataset_name)


if __name__ == "__main__":
    main()
