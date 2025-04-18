from config import settings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


def chunk_text(text: str) -> list[str]:
    if settings.EMBEDDING_MODEL_PROVIDER == "huggingface":
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"], chunk_size=500, chunk_overlap=0
        )
        text_split = character_splitter.split_text(text)
        token_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=50,
            tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
            model_name=settings.EMBEDDING_MODEL_ID,
        )
        chunks = []

        for section in text_split:
            chunks.extend(token_splitter.split_text(section))
    elif settings.EMBEDDING_MODEL_PROVIDER == "openai":
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=settings.EMBEDDING_MODEL_ID,
            chunk_size=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
            chunk_overlap=50,
        )
        chunks = text_splitter.split_text(text)
    else:
        raise ValueError(
            f"Invalid embedding model provider: {settings.EMBEDDING_MODEL_PROVIDER}"
        )
    return chunks
