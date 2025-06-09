from abc import ABC, abstractmethod

from langchain_core.prompts import PromptTemplate


class ChatbotBase(ABC):
    """Abstract base class for all chatbots."""

    @abstractmethod
    def generate(
        self,
        query: str,
        enable_rag: bool = False,
        sample_for_evaluation_rate: float = 0.0,
        dataset_name: str = "PharmAssistMonitoringDataset",
    ) -> dict:
        """
        Generates a response for the given query.

        Args:
            query: The query to generate a response for.
            enable_rag: Whether to enable RAG.
            sample_for_evaluation_rate: The rate at which to sample for evaluation.
            dataset_name: The name of the dataset to save samples to.

        Returns:
            A dictionary containing the response and the evaluation metrics.
        """
        pass

    @abstractmethod
    def call_llm_service(self, messages: list[dict[str, str]]) -> str:
        """
        Calls the LLM service to generate a response for the given query.
        """
        pass

    @abstractmethod
    def format_prompt(
        self,
        system_prompt,
        prompt_template: PromptTemplate,
        prompt_template_variables: dict,
    ) -> tuple[list[dict[str, str]], int]:
        """
        Formats the prompt for the given query.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """
        Returns the configuration of the chatbot.
        """
        pass
