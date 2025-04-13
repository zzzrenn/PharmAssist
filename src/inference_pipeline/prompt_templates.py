from langchain.prompts import PromptTemplate

from core.rag.prompt_templates import BasePromptTemplate


class InferenceTemplate(BasePromptTemplate):
    simple_system_prompt: str = """
    You are an AI language model assistant. Your task is to generate a cohesive and concise response based on the user's instruction by using a similar writing style and voice.
"""
    simple_prompt_template: str = """
### Instruction:
{question}
"""

    rag_system_prompt: str = """You are a clinical guideline assistant for pharmacists. Respond ONLY using exact recommendations from provided documents.
    **Critical Rules**:
    - Preserve exact numerical values
    - Highlight 'Do not'/'Contraindicated' statements first
    - Include document update date
    - Use chapter titles verbatim from source

    """
    rag_prompt_template: str = """
### Instruction:
{question}

### Context:
{context}
"""

    def create_template(self, enable_rag: bool = True) -> tuple[str, PromptTemplate]:
        if enable_rag is True:
            return self.rag_system_prompt, PromptTemplate(
                template=self.rag_prompt_template,
                input_variables=["question", "context"],
            )

        return self.simple_system_prompt, PromptTemplate(
            template=self.simple_prompt_template, input_variables=["question"]
        )
