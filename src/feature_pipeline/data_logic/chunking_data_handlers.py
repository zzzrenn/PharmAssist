import hashlib
from abc import ABC, abstractmethod

from models.base import DataModel
from models.chunk import NiceChunkModel
from models.clean import NiceCleanedModel
from utils.chunking import chunk_text


class ChunkingDataHandler(ABC):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: DataModel) -> list[DataModel]:
        pass


class NiceChunkingHandler(ChunkingDataHandler):
    def chunk(self, data_model: NiceCleanedModel) -> list[NiceChunkModel]:
        data_models_list = []

        text_content = data_model.cleaned_content
        chunks = chunk_text(text_content)

        for chunk in chunks:
            model = NiceChunkModel(
                entry_id=data_model.entry_id,
                chunk_id=hashlib.md5(chunk.encode()).hexdigest(),
                title=data_model.title,
                url=data_model.url,
                last_updated=data_model.last_updated,
                chunk_content=chunk,
                type=data_model.type,
            )
            data_models_list.append(model)

        return data_models_list