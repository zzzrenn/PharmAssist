import argparse

from chatbots import chatbot
from config import settings
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, LevenshteinRatio, Moderation

from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts

logger = get_logger(__name__)


class Evaluator:
    def __init__(self):
        """Initialize the evaluator with a chatbot instance."""
        self.chatbot = chatbot

    def evaluation_task(self, x: dict) -> dict:
        """Evaluation task that uses the stored chatbot instance."""
        result = self.chatbot.generate(
            query=x["instruction"],
            enable_rag=False,
        )
        answer = result["answer"]

        return {
            "input": x["instruction"],
            "output": answer,
            "expected_output": x["content"],
            "reference": x["content"],
        }

    def run_evaluation(self, dataset_name: str = "PharmAssistTestDataset") -> None:
        """Run the evaluation on the specified dataset."""
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
            **self.chatbot.get_config(),
            "embedding_model_id": settings.EMBEDDING_MODEL_ID,
        }
        scoring_metrics = [
            LevenshteinRatio(),
            Hallucination(model="gpt-4o-mini"),
            Moderation(model="gpt-4o-mini"),
        ]
        evaluate(
            dataset=dataset,
            task=self.evaluation_task,
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            task_threads=1,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM on test dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PharmAssistTestDataset",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.run_evaluation(dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
