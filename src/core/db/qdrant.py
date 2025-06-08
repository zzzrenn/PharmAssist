from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Batch,
    Distance,
    Fusion,
    Modifier,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

import core.logger_utils as logger_utils
from core.config import settings

logger = logger_utils.get_logger(__name__)


class QdrantDatabaseConnector:
    _instance: QdrantClient | None = None

    def __init__(self) -> None:
        if self._instance is None:
            if settings.USE_QDRANT_CLOUD:
                self._instance = QdrantClient(
                    url=settings.QDRANT_CLOUD_URL,
                    api_key=settings.QDRANT_APIKEY,
                )
            else:
                self._instance = QdrantClient(
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                )

    def get_collection(self, collection_name: str):
        return self._instance.get_collection(collection_name=collection_name)

    def create_non_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name, vectors_config={}
        )

    def create_vector_collection(self, collection_name: str):
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=settings.EMBEDDING_SIZE, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    modifier=Modifier.IDF,  # Use IDF modifier for BM25
                )
            },
        )

    def write_data(self, collection_name: str, points: Batch):
        try:
            self._instance.upsert(collection_name=collection_name, points=points)
        except Exception:
            logger.exception("An error occurred while inserting data.")

            raise

    def search(
        self,
        collection_name: str,
        query_vector: list,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        return self._instance.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points

    def hybrid_search_rrf(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_vector: dict,
        query_filter: models.Filter | None = None,
        limit: int = 3,
    ) -> list:
        """
        Perform hybrid search using both dense and sparse vectors with Reciprocal Rank Fusion (RRF).

        Args:
            collection_name: Name of the collection to search
            dense_vector: Dense embedding vector
            sparse_vector: Sparse vector (e.g., from BM25)
            query_filter: Optional filter to apply
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        # Create prefetch queries for each vector type
        prefetch_queries = [
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=limit,
            )
        ]

        # Add sparse vector prefetch if provided
        if sparse_vector:
            prefetch_queries.append(
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit,
                )
            )

        return self._instance.query_points(
            collection_name=collection_name,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
        ).points

    def scroll(self, collection_name: str, limit: int):
        return self._instance.scroll(collection_name=collection_name, limit=limit)

    def close(self):
        if self._instance:
            self._instance.close()

            logger.info("Connected to database has been closed.")
