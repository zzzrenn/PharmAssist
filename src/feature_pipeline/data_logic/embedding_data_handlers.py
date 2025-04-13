from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import NiceChunkModel
from models.embedded_chunk import NiceEmbeddedChunkModel
from utils.embeddings import embedd_text


class EmbeddingDataHandler(ABC):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    @abstractmethod
    def embedd(self, data_model: DataModel) -> DataModel:
        pass


class NiceEmbeddingHandler(EmbeddingDataHandler):
    def embedd(self, data_model: NiceChunkModel) -> NiceEmbeddedChunkModel:
        return NiceEmbeddedChunkModel(
            entry_id=data_model.entry_id,
            chunk_id=data_model.chunk_id,
            title=data_model.title,
            url=data_model.url,
            last_updated=data_model.last_updated,
            chunk_content=data_model.chunk_content,
            embedded_content=embedd_text(data_model.chunk_content),
            type=data_model.type,
        )
