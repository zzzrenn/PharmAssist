import concurrent.futures

import opik
import torch
from qdrant_client import models

import core.logger_utils as logger_utils
from core import lib
from core.config import settings
from core.db.qdrant import QdrantDatabaseConnector
from core.models.embeddings import embedding_model_factory
from core.rag.query_expansion import QueryExpansion
from core.rag.reranking import Reranker
from core.rag.self_query import SelfQuery

logger = logger_utils.get_logger(__name__)


class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion, self-query, hybrid search and reranking.
    """

    def __init__(
        self,
        hybrid_search: bool = False,
        self_query: bool = False,
        n_query_expansion: int = 0,
        rerank: bool = False,
    ) -> None:
        self._client = QdrantDatabaseConnector()
        device = settings.EMBEDDING_MODEL_DEVICE
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
            logger.warning(
                "CUDA device requested but not available. Falling back to CPU."
            )
        else:
            logger.info("Using CUDA device for embeddings.")
        self._dense_embedder = embedding_model_factory.create_embedding_model()
        self._sparse_embedder = (
            embedding_model_factory.create_embedding_model(sparse=True)
            if hybrid_search
            else None
        )
        self._query_expander = QueryExpansion() if n_query_expansion > 0 else None
        self.n_query_expansion = n_query_expansion
        self._metadata_extractor = SelfQuery() if self_query else None
        self._reranker = Reranker() if rerank else None

        logger.info(
            "Retriever initialized successfully.",
            hybrid_search=hybrid_search,
            self_query=self_query,
            n_query_expansion=n_query_expansion,
            rerank=rerank,
        )

    def _search_single_query(self, generated_query: str, chapter_name: str, k: int):
        assert k > 1, "k should be greater than 1"

        dense_query_vector = self._dense_embedder.embed(generated_query)
        sparse_query_vector = (
            self._sparse_embedder.embed(generated_query).as_object()
            if self._sparse_embedder
            else None
        )

        vectors = [
            self._client.hybrid_search_rrf(
                collection_name="vector_nice",
                query_filter=models.Filter(
                    must=(
                        [
                            models.FieldCondition(
                                key="chapter",
                                match=models.MatchValue(
                                    value=chapter_name,
                                ),
                            )
                        ]
                        if chapter_name
                        else None
                    )
                ),
                dense_vector=dense_query_vector,
                sparse_vector=sparse_query_vector,
                limit=k,
            )
        ]

        return lib.flatten(vectors)

    @opik.track(name="retriever.retrieve_top_k")
    def retrieve_top_k(self, query: str, k: int) -> list:
        generated_queries = [query]

        # query expansion
        if self.n_query_expansion > 0:
            generated_queries.extend(
                self._query_expander.generate_response(
                    query, to_expand_to_n=self.n_query_expansion
                )
            )

        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(generated_queries),
        )

        chapter_name = None
        if self._metadata_extractor:
            chapter_name = self._metadata_extractor.generate_response(query)
            if chapter_name:
                logger.info(
                    "Successfully extracted the chapter name from the query.",
                    chapter_name=chapter_name,
                )
            else:
                logger.warning("Did not found any chapter name in the user's prompt.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, query, chapter_name, k)
                for query in generated_queries
            ]

            hits = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            hits = lib.flatten(hits)

        logger.info("All documents retrieved successfully.", num_documents=len(hits))

        return hits

    @opik.track(name="retriever.rerank")
    def rerank(self, query: str, hits: list, keep_top_k: int) -> list[str]:
        if not self._reranker:
            logger.info("Reranking is disabled, returning original hits")
            return [hit.payload["content"] for hit in hits[:keep_top_k]]

        content_list = [hit.payload["content"] for hit in hits]
        rerank_hits = self._reranker.generate_response(
            query=query, passages=content_list, keep_top_k=keep_top_k
        )

        logger.info("Documents reranked successfully.", num_documents=len(rerank_hits))

        return rerank_hits
