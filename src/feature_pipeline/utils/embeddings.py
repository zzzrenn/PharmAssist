from InstructorEmbedding import INSTRUCTOR
from sentence_transformers.SentenceTransformer import SentenceTransformer

from config import settings


def embedd_text(text: str):
    model = SentenceTransformer(settings.EMBEDDING_MODEL_ID, device=settings.EMBEDDING_MODEL_DEVICE)
    return model.encode(text)