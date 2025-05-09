import pprint

import opik
import torch
from config import settings
from langchain.prompts import PromptTemplate
from opik import opik_context
from prompt_templates import InferenceTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import compute_num_tokens, truncate_text_to_max_tokens

from core import logger_utils
from core.opik_utils import add_to_dataset_with_sampling
from core.rag.retriever import VectorRetriever

logger = logger_utils.get_logger(__name__)


class Chatbot:
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        if not mock:
            # Initialize model from Hugging Face using config
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_ID, trust_remote_code=True
            )
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            device = settings.MODEL_DEVICE
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
                logger.warning(
                    "CUDA device requested but not available. Falling back to CPU."
                )
            else:
                logger.info("Using CUDA device for LLM.")

            # Configure bitsandbytes quantization only for CUDA
            if device.startswith("cuda"):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": device,
                    "trust_remote_code": True,
                }
                logger.info("Using bitsandbytes quantization for LLM.")
            else:
                model_kwargs = {
                    "device_map": device,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                }

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_ID, **model_kwargs
            )
            self.model.eval()  # Set to evaluation mode
        self.prompt_template_builder = InferenceTemplate()

    @opik.track(name="inference_pipeline.generate")
    def generate(
        self,
        query: str,
        enable_rag: bool = False,
        sample_for_evaluation: bool = False,
    ) -> dict:
        system_prompt, prompt_template = self.prompt_template_builder.create_template(
            enable_rag=enable_rag
        )
        prompt_template_variables = {"question": query}

        if enable_rag is True:
            retriever = VectorRetriever(query=query)
            hits = retriever.retrieve_top_k(
                k=settings.TOP_K, to_expand_to_n_queries=settings.EXPAND_N_QUERY
            )
            context = retriever.rerank(hits=hits, keep_top_k=settings.KEEP_TOP_K)
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
            tags=["rag"],
            metadata={
                "prompt_template": prompt_template.template,
                "prompt_template_variables": prompt_template_variables,
                "model_id": settings.MODEL_ID,
                "embedding_model_id": settings.EMBEDDING_MODEL_ID,
                "input_tokens": input_num_tokens,
                "answer_tokens": num_answer_tokens,
                "total_tokens": input_num_tokens + num_answer_tokens,
            },
        )

        answer = {"answer": answer, "context": context}
        if sample_for_evaluation is True:
            add_to_dataset_with_sampling(
                item={"input": {"query": query}, "expected_output": answer},
                dataset_name="PharmAssistMonitoringDataset",
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

        # Format messages according to Qwen's chat template
        formatted_messages = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the input without padding
        inputs = self.tokenizer(
            formatted_messages,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_INPUT_TOKENS,
        )

        # Move input tensors to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=settings.MAX_TOTAL_TOKENS - settings.MAX_INPUT_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,  # Enable KV cache for faster generation
            )

        # Decode the response and remove the input prompt
        answer = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        answer = answer.strip()

        return answer
