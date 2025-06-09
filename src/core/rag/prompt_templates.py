from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from pydantic import BaseModel


class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self, *args) -> PromptTemplate:
        pass


class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to generate exactly {to_expand_to_n}
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
    prompt: str = """You are an AI assistant designed to extract chapter from questions about pharmaceutical guidelines.
    The possible chapters are:
    - Overview
    - Recommendations
    - Recommendations for research
    - Rationale and impact
    - Context
    - Finding more information and committee details
    - Update information

    Your response should consists of only the extracted chapter. If no chapter is found, return "none".

    ### Examples:
    QUESTION 1:
    individualised care approach for type 2 diabetes
    RESPONSE 1:
    Recommendations

    QUESTION 2:
    Reasoning behind diagnosis recommendations of hypertension update
    RESPONSE 2:
    Rationale and impact


    QUESTION 3:
    Update date for periodontitis information
    RESPONSE 3:
    Update information

    QUESTION 4:
    What is the probability of side effects of paracetamol?
    RESPONSE 4:
    none

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
