import pprint

import opik
from config import settings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from opik import opik_context
from opik.integrations.langchain import OpikTracer
from prompt_templates import InferenceTemplate
from utils import compute_num_tokens, truncate_text_to_max_tokens

from core import logger_utils
from core.opik_utils import add_to_dataset_with_sampling
from core.rag.retriever import VectorRetriever
from inference_pipeline.chatbots.chatbot_base import ChatbotBase

logger = logger_utils.get_logger(__name__)


class ChatbotOpenAI(ChatbotBase):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        if not mock:
            # Initialize OpenAI model using LangChain
            self.model = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
                max_tokens=settings.MAX_TOTAL_TOKENS - settings.MAX_INPUT_TOKENS,
            )
            logger.info(f"Initialized OpenAI model: {settings.OPENAI_MODEL_ID}")

        self.prompt_template_builder = InferenceTemplate()
        self.retriever = VectorRetriever(
            hybrid_search=settings.ENABLE_SPARSE_EMBEDDING,
            self_query=settings.ENABLE_SELF_QUERY,
            n_query_expansion=settings.EXPAND_N_QUERY,
            rerank=settings.ENABLE_RERANKING,
        )

        # Initialize Opik tracer for LangChain integration
        self.opik_tracer = OpikTracer(tags=["openai_chatbot"])

    @opik.track(name="inference_pipeline.generate")
    def generate(
        self,
        query: str,
        enable_rag: bool = False,
        sample_for_evaluation_rate: float = 0.0,
        dataset_name: str = "PharmAssistMonitoringDataset",
    ) -> dict:
        assert 0 <= sample_for_evaluation_rate <= 1, (
            "Sample rate must be between 0 and 1"
        )

        system_prompt, prompt_template = self.prompt_template_builder.create_template(
            enable_rag=enable_rag
        )
        prompt_template_variables = {"question": query}

        if enable_rag is True:
            hits = self.retriever.retrieve_top_k(query=query, k=settings.TOP_K)
            context = self.retriever.rerank(
                query=query, hits=hits, keep_top_k=settings.KEEP_TOP_K
            )
            prompt_template_variables["context"] = context
        else:
            context = None

        messages, input_num_tokens = self.format_prompt(
            system_prompt, prompt_template, prompt_template_variables
        )

        logger.debug(f"Prompt: {pprint.pformat(messages)}")
        answer = self.call_llm_service(messages=messages)
        logger.debug(f"Answer: {answer}")

        num_answer_tokens = compute_num_tokens(answer)
        opik_context.update_current_trace(
            tags=["rag", "openai"],
            metadata={
                "prompt_template": prompt_template.template,
                "prompt_template_variables": prompt_template_variables,
                "model_id": settings.OPENAI_MODEL_ID,
                "embedding_model_id": settings.EMBEDDING_MODEL_ID,
                "input_tokens": input_num_tokens,
                "answer_tokens": num_answer_tokens,
                "total_tokens": input_num_tokens + num_answer_tokens,
            },
        )

        answer = {"answer": answer, "context": context}
        if sample_for_evaluation_rate:
            add_to_dataset_with_sampling(
                item={"input": {"query": query}, "expected_output": answer},
                dataset_name=dataset_name,
                sample_rate=sample_for_evaluation_rate,
            )

        return answer

    @opik.track(name="inference_pipeline.format_prompt")
    def format_prompt(
        self,
        system_prompt,
        prompt_template: PromptTemplate,
        prompt_template_variables: dict,
    ) -> tuple[list[dict[str, str]], int]:
        prompt = prompt_template.format(**prompt_template_variables)

        num_system_prompt_tokens = compute_num_tokens(system_prompt)
        prompt, prompt_num_tokens = truncate_text_to_max_tokens(
            prompt, max_tokens=settings.MAX_INPUT_TOKENS - num_system_prompt_tokens
        )
        total_input_tokens = num_system_prompt_tokens + prompt_num_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        return messages, total_input_tokens

    @opik.track(name="inference_pipeline.call_llm_service")
    def call_llm_service(self, messages: list[dict[str, str]]) -> str:
        if self._mock is True:
            logger.warning("Mocking LLM service call.")
            return "Mocked answer."

        try:
            # Convert messages to LangChain format
            from langchain.schema import HumanMessage, SystemMessage

            langchain_messages = []
            for message in messages:
                if message["role"] == "system":
                    langchain_messages.append(SystemMessage(content=message["content"]))
                elif message["role"] == "user":
                    langchain_messages.append(HumanMessage(content=message["content"]))

            # Call OpenAI model with Opik tracing
            response = self.model.invoke(
                langchain_messages, config={"callbacks": [self.opik_tracer]}
            )

            answer = response.content.strip()
            logger.info("Successfully generated response using OpenAI model.")

            return answer

        except Exception as e:
            logger.error(f"Error calling OpenAI service: {str(e)}")
            raise e

    def get_config(self) -> dict:
        return {
            "provider": "openai",
            "model_id": settings.OPENAI_MODEL_ID,
        }
