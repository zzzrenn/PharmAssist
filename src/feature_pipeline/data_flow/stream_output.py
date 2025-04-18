from bytewax.outputs import DynamicSink, StatelessSinkPartition
from models.base import VectorDBDataModel
from qdrant_client.models import Batch, PointIdsList

from core import get_logger
from core.db.qdrant import QdrantDatabaseConnector

logger = get_logger(__name__)


class QdrantOutput(DynamicSink):
    """
    Bytewax class that facilitates the connection to a Qdrant vector DB.
    Inherits DynamicSink because of the ability to create different sink sources (e.g, vector and non-vector collections)
    """

    def __init__(self, connection: QdrantDatabaseConnector, sink_type: str):
        self._connection = connection
        self._sink_type = sink_type

        collections = {
            "cleaned_nice": False,
            "vector_nice": True,
            "cleaned_test": False,
            "vector_test": True,
        }

        for collection_name, is_vector in collections.items():
            try:
                self._connection.get_collection(collection_name=collection_name)
            except Exception:
                logger.warning(
                    "Couldn't access the collection. Creating a new one...",
                    collection_name=collection_name,
                )

                if is_vector:
                    self._connection.create_vector_collection(
                        collection_name=collection_name
                    )
                else:
                    self._connection.create_non_vector_collection(
                        collection_name=collection_name
                    )

    def build(
        self, step_id: str, worker_index: int, worker_count: int
    ) -> StatelessSinkPartition:
        if self._sink_type == "clean":
            return QdrantCleanedDataSink(connection=self._connection)
        elif self._sink_type == "vector":
            return QdrantVectorDataSink(connection=self._connection)
        else:
            raise ValueError(f"Unsupported sink type: {self._sink_type}")


class QdrantCleanedDataSink(StatelessSinkPartition):
    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        payloads = [item.to_payload() for item in items]
        ids, data = zip(*payloads)
        collection_name = get_clean_collection(data_type=data[0]["type"])
        entry_id = data[0]["entry_id"]

        # Find and delete existing vectors by id in metadata
        existing_points = self._client._instance.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "entry_id", "match": {"value": entry_id}}]},
            limit=10000,
        )[0]

        if existing_points:
            qdrant_ids = [point.id for point in existing_points]
            self._client._instance.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=qdrant_ids),
            )
            logger.info(
                "Deleted cleaned data",
                collection_name=collection_name,
                num=len(qdrant_ids),
                entry_id=entry_id,
            )

        # Insert new points
        self._client.write_data(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors={}, payloads=data),
        )

        logger.info(
            "Successfully inserted cleaned data",
            collection_name=collection_name,
            num=len(ids),
            entry_id=entry_id,
        )


class QdrantVectorDataSink(StatelessSinkPartition):
    def __init__(self, connection: QdrantDatabaseConnector):
        self._client = connection

    def write_batch(self, items: list[VectorDBDataModel]) -> None:
        payloads = [item.to_payload() for item in items]
        ids, vectors, meta_data = zip(*payloads)
        collection_name = get_vector_collection(data_type=meta_data[0]["type"])
        match_id = meta_data[0]["id"]

        # Find and delete existing vectors by id in metadata
        existing_points = self._client._instance.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "id", "match": {"value": match_id}}]},
            limit=10000,
        )[0]

        if existing_points:
            qdrant_ids = [point.id for point in existing_points]
            self._client._instance.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=qdrant_ids),
            )
            logger.info(
                "Deleted existing vector point(s)",
                collection_name=collection_name,
                num=len(qdrant_ids),
                id=match_id,
            )

        # Insert new points
        self._client.write_data(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=meta_data),
        )

        logger.info(
            "Successfully inserted vector point(s)",
            collection_name=collection_name,
            num=len(ids),
            id=match_id,
        )


def get_clean_collection(data_type: str) -> str:
    if data_type == "NICE_GUIDELINE":
        return "cleaned_nice"
    elif data_type == "test_collection":
        return "cleaned_test"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_vector_collection(data_type: str) -> str:
    if data_type == "NICE_GUIDELINE":
        return "vector_nice"
    elif data_type == "test_collection":
        return "vector_test"
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
