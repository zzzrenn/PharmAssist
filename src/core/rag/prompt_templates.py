from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from pydantic import BaseModel


class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self, *args) -> PromptTemplate:
        pass


class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to generate {to_expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "to_expand_to_n": to_expand_to_n,
            },
        )


class SelfQueryTemplate(BasePromptTemplate):
    prompt: str = """You are an AI assistant designed to extract metadata from questions about clinical guidelines.
    The required information to extract includes:
    1. **Section type** (e.g., Recommendations, Context, Research, Implementation Guidance).
    2. **Entities** (e.g., drugs, conditions, procedures).
    3. **Demographics** (e.g., adults, elderly, pregnancy).
    4. **Date filters** (e.g., last_updated).
    5. **Guideline title** (e.g., "Chronic heart failure in adults: diagnosis and management").

    Your response must be a JSON object with two keys: `search_terms` (list of keywords) and `filters` (key-value pairs for metadata).
    If no relevant metadata is found, return `"none"` for both keys.

    ### Examples:
    QUESTION 1:
    "What are the recommendations for beta-blockers in adults with heart failure?"
    RESPONSE 1:
    {
    "search_terms": ["beta-blockers", "recommendations"],
    "filters": {
        "section_type": ["Clinical Recommendations"],
        "entities": ["beta-blockers", "heart failure"],
        "demographics": ["adults"]
        }
    }

    QUESTION 2:
    "Show tools for implementing heart failure guidelines."
    RESPONSE 2:
    {
    "search_terms": ["tools", "implementing"],
    "filters": {
        "section_type": ["Implementation Guidance"],
        "entities": ["heart failure"]
        }
    }

    QUESTION 3:
    "I want general information about heart failure."
    RESPONSE 3:
    {
    "search_terms": ["general information"],
    "filters": {
        "section_type": ["Overview", "Context"]
        }
    }

    ### Notes:
    - Use synonyms for section types (e.g., "guidance" to "Implementation Guidance").
    - Ignore non-clinical terms (e.g., "tools" to filter by section_type, not entities).
    - If no filters apply, return:
    {
    "search_terms": "none",
    "filters": "none"
    }

    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class RerankingTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to rerank passages related to a query
    based on their relevance.
    The most relevant passages should be put at the beginning.
    You should only pick at max {keep_top_k} passages.
    The provided and reranked documents are separated by '{separator}'.

    The following are passages related to this query: {question}.

    Passages:
    {passages}
    """

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],
            partial_variables={"keep_top_k": keep_top_k, "separator": self.separator},
        )

    @property
    def separator(self) -> str:
        return "\n#next-document#\n"
