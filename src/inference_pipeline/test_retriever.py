import sys
import os
# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


from core import get_logger
from core.rag.retriever import VectorRetriever

logger = get_logger(__name__)

if __name__ == "__main__":
    query = """
        Give me first treatment for diabetes type 2.
        """

    retriever = VectorRetriever(query=query)
    hits = retriever.retrieve_top_k(k=6, to_expand_to_n_queries=5)
    reranked_hits = retriever.rerank(hits=hits, keep_top_k=5)

    logger.info("====== RETRIEVED DOCUMENTS ======")
    for rank, hit in enumerate(reranked_hits):
        logger.info(f"Rank = {rank} : {hit}")