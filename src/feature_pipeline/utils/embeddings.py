from core.logger_utils import get_logger
from core.models.embeddings import embedding_model_factory

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
        self._dense_model = embedding_model_factory.create_embedding_model()
        self._sparse_model = embedding_model_factory.create_embedding_model(sparse=True)

    def encode(self, text: str, sparse: bool = False):
        if sparse:
            if self._sparse_model:
                return self._sparse_model.embed(text).as_object()
            else:
                return None
        else:
            return self._dense_model.embed(text)


def embedd_text(text: str, sparse: bool = False):
    model_manager = EmbeddingModelManager()
    return model_manager.encode(text, sparse)
