from typing import List

from data_logic.chunking_data_handlers import (
    ChunkingDataHandler,
    NiceChunkingHandler,
)
from data_logic.cleaning_data_handlers import (
    CleaningDataHandler,
    NiceCleaningHandler,
)
from data_logic.embedding_data_handlers import (
    EmbeddingDataHandler,
    NiceEmbeddingHandler,
)
from models.base import DataModel
from models.raw import NiceRawModel

from core import get_logger

logger = get_logger(__name__)


class RawDispatcher:
    @staticmethod
    def handle_mq_message(message: dict) -> DataModel:
        data_type = message.get("type")

        logger.info("Received message.", data_type=data_type)

        if data_type == "NICE_GUIDELINE":
            return NiceRawModel(**message)
        elif data_type == "test_collection":
            return NiceRawModel(**message)
        else:
            raise ValueError("Unsupported data type")


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> CleaningDataHandler:
        if data_type == "NICE_GUIDELINE":
            return NiceCleaningHandler()
        elif data_type == "test_collection":
            return NiceCleaningHandler()
        else:
            raise ValueError("Unsupported data type")


class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory()

    @classmethod
    def dispatch_cleaner(cls, data_model: DataModel) -> List[DataModel]:
        data_type = data_model.type
        handler = cls.cleaning_factory.create_handler(data_type)
        clean_models_list = handler.clean(data_model)

        logger.info(
            "Data cleaned successfully into multiple models.",
            data_type=data_type,
            num_cleaned_models=len(clean_models_list),
        )

        return clean_models_list


class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> ChunkingDataHandler:
        if data_type == "NICE_GUIDELINE":
            return NiceChunkingHandler()
        elif data_type == "test_collection":
            return NiceChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    chunking_factory = ChunkingHandlerFactory

    @classmethod
    def dispatch_chunker(cls, data_model: DataModel) -> list[DataModel]:
        data_type = data_model.type
        handler = cls.chunking_factory.create_handler(data_type)
        chunk_models = handler.chunk(data_model)

        logger.info(
            "Cleaned content chunked successfully.",
            num=len(chunk_models),
            data_type=data_type,
        )

        return chunk_models


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_type) -> EmbeddingDataHandler:
        if data_type == "NICE_GUIDELINE":
            return NiceEmbeddingHandler()
        elif data_type == "test_collection":
            return NiceEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    embedding_factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch_embedder(cls, data_model: DataModel) -> DataModel:
        data_type = data_model.type
        handler = cls.embedding_factory.create_handler(data_type)
        embedded_chunk_model = handler.embedd(data_model)

        logger.info(
            "Chunk embedded successfully.",
            data_type=data_type,
            dense_embedding_len=len(embedded_chunk_model.dense_embedded_content),
            sparse_embedding=True
            if embedded_chunk_model.sparse_embedded_content
            else False,
        )

        return embedded_chunk_model
