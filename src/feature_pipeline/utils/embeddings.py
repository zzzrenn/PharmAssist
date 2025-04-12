from InstructorEmbedding import INSTRUCTOR
from sentence_transformers.SentenceTransformer import SentenceTransformer
import torch

from config import settings
from core.logger_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModelManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModelManager, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        device = settings.EMBEDDING_MODEL_DEVICE
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA device requested but not available. Falling back to CPU.")
        else:
            logger.info("Using CUDA device for embeddings.")
        
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL_ID, device=device)

    def encode(self, text: str):
        return self._model.encode(text)


def embedd_text(text: str):
    model_manager = EmbeddingModelManager()
    return model_manager.encode(text)