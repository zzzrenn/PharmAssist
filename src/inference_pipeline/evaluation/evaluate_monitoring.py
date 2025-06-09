import argparse

import opik
from chatbots import chatbot
from config import settings
from opik.evaluation import evaluate
from opik.evaluation.metrics import AnswerRelevance, Hallucination, Moderation

from core.logger_utils import get_logger


class MonitoringEvaluator:
    def __init__(self):
        """Initialize the MonitoringEvaluator with a chatbot instance.

        Args:
            chatbot: The chatbot/RAG system to be evaluated
        """
        self.logger = get_logger(__name__)
        self.chatbot = chatbot
        self.client = opik.Opik()

    def evaluation_task(self, x: dict) -> dict:
        """Transform dataset item for evaluation."""
        return {
            "input": x["input"]["query"],
            "context": x["expected_output"]["context"],
            "output": x["expected_output"]["answer"],
        }

    def get_dataset(self, dataset_name: str):
        """Retrieve dataset from Opik."""
        try:
            dataset = self.client.get_dataset(dataset_name)
            return dataset
        except Exception:
            self.logger.error(
                f"Monitoring dataset '{dataset_name}' not found in Opik. Exiting."
            )
            exit(1)

    def run_evaluation(self, dataset_name: str = "PharmAssistMonitoringDataset"):
        """Run the evaluation on the specified dataset."""
        self.logger.info(f"Evaluating Opik dataset: '{dataset_name}'")

        dataset = self.get_dataset(dataset_name)

        experiment_config = {
            **self.chatbot.get_config(),
            "embedding_model_id": settings.EMBEDDING_MODEL_ID,
            "hybrid_search": settings.ENABLE_SPARSE_EMBEDDING,
            "self_query": settings.ENABLE_SELF_QUERY,
            "expand_n_query": settings.EXPAND_N_QUERY,
            "top_k": settings.TOP_K,
            "keep_top_k": settings.KEEP_TOP_K,
            "rerank": settings.ENABLE_RERANKING,
        }

        scoring_metrics = [
            Hallucination(model="gpt-4o-mini"),
            Moderation(model="gpt-4o-mini"),
            AnswerRelevance(model="gpt-4o-mini"),
        ]
        evaluate(
            dataset=dataset,
            task=self.evaluation_task,
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            task_threads=1,
        )


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

    # Initialize evaluator (chatbot can be passed here when available)
    evaluator = MonitoringEvaluator(chatbot=None)

    # Run evaluation
    evaluator.run_evaluation(dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
