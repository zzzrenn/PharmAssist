import argparse

from chatbots import chatbot
from config import settings
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    ContextPrecision,
    ContextRecall,
    Hallucination,
)

from core.logger_utils import get_logger
from core.opik_utils import create_dataset_from_artifacts

logger = get_logger(__name__)


class RAGEvaluator:
    def __init__(self):
        self.chatbot = chatbot

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
            AnswerRelevance(model="gpt-4o-mini"),
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
            nb_samples=10,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG on test dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PharmAssistTestDataset_v2",
        help="Name of the dataset to evaluate",
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name

    evaluator = RAGEvaluator()
    evaluator.evaluate_dataset(dataset_name)


if __name__ == "__main__":
    main()
