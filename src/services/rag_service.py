"""
RAG (Retrieval-Augmented Generation) service for search, ranking, and retrieval operations.

This module provides the RagService class that orchestrates document and code search,
hybrid search functionality, result fusion, and CrossEncoder-based re-ranking.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

# Import clients and services
try:
    from ..clients.qdrant_client import QdrantClientWrapper
    from ..services.embedding_service import create_embedding
    from ..device_manager import cleanup_gpu_memory
except ImportError:
    from clients.qdrant_client import QdrantClientWrapper
    from services.embedding_service import create_embedding
    from device_manager import cleanup_gpu_memory

logger = logging.getLogger(__name__)


class RagService:
    """
    Service class for Retrieval-Augmented Generation operations.

    Handles document search, code search, hybrid search functionality,
    result fusion, and CrossEncoder-based re-ranking.
    """

    def __init__(
        self,
        qdrant_client: QdrantClientWrapper,
        reranking_model: Optional[CrossEncoder] = None,
    ):
        """
        Initialize the RAG service.

        Args:
            qdrant_client: Qdrant client wrapper for vector database operations
            reranking_model: Optional CrossEncoder model for result re-ranking
        """
        self.qdrant_client = qdrant_client
        self.reranking_model = reranking_model
        self.use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        self.use_hybrid_search = (
            os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
        )

    def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using Qdrant vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of matching documents
        """
        # Create embedding for the query
        query_embedding = create_embedding(query)

        # Execute the search using Qdrant client
        try:
            results = self.qdrant_client.search_documents(
                query_embedding=query_embedding,
                match_count=match_count,
                filter_metadata=filter_metadata,
            )
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def search_code_examples(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples using Qdrant vector similarity.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results

        Returns:
            List of matching code examples
        """
        # Create a more descriptive query for better embedding match
        enhanced_query = (
            f"Code example for {query}\n\nSummary: Example code showing {query}"
        )

        # Create embedding for the enhanced query
        query_embedding = create_embedding(enhanced_query)

        # Execute the search using Qdrant client
        try:
            results = self.qdrant_client.search_code_examples(
                query_embedding=query_embedding,
                match_count=match_count,
                filter_metadata=filter_metadata,
                source_filter=source_id,
            )
            return results
        except Exception as e:
            logger.error(f"Error searching code examples: {e}")
            return []

    def update_source_info(self, source_id: str, summary: str, word_count: int):
        """
        Update source information using Qdrant client wrapper.

        Args:
            source_id: The source ID (domain)
            summary: Summary of the source
            word_count: Total word count for the source
        """
        try:
            self.qdrant_client.update_source_info(source_id, summary, word_count)
        except Exception as e:
            logger.error(f"Error updating source {source_id}: {e}")

    def hybrid_search(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "documents",
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse search results.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            search_type: Type of search - "documents" or "code_examples"

        Returns:
            List of search results with fused scores
        """
        if not self.use_hybrid_search:
            # Fallback to regular search if hybrid search is disabled
            if search_type == "code_examples":
                return self.search_code_examples(query, match_count, filter_metadata)
            else:
                return self.search_documents(query, match_count, filter_metadata)

        try:
            if search_type == "code_examples":
                results = self.qdrant_client.hybrid_search_code_examples(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                )
            else:
                results = self.qdrant_client.hybrid_search_documents(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                )

            logger.debug(f"Hybrid search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to regular search
            if search_type == "code_examples":
                return self.search_code_examples(query, match_count, filter_metadata)
            else:
                return self.search_documents(query, match_count, filter_metadata)

    def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.

        Args:
            query: The search query
            results: List of search results
            content_key: The key in each result dict that contains the text content

        Returns:
            Reranked list of results
        """
        if not self.reranking_model or not results or not self.use_reranking:
            return results

        try:
            logger.debug(
                f"Starting reranking with {len(results)} results for query: '{query[:50]}...'"
            )

            # Extract content from results
            texts = [result.get(content_key, "") for result in results]

            # Create pairs of [query, document] for the cross-encoder
            pairs = [[query, text] for text in texts]
            logger.debug(f"Created {len(pairs)} query-document pairs for reranking")

            # Get relevance scores from the cross-encoder
            scores = self.reranking_model.predict(pairs)
            logger.debug(f"Generated rerank scores: {[f'{s:.4f}' for s in scores[:5]]}")

            # Add scores to results and sort by score (descending)
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])

            # Sort by rerank score
            reranked = sorted(
                results, key=lambda x: x.get("rerank_score", 0), reverse=True
            )
            logger.debug(
                f"Reranked results - top score: {reranked[0].get('rerank_score', 0):.4f}"
            )

            # Clean up GPU memory after processing (critical for long-running processes)
            cleanup_gpu_memory()

            return reranked
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results

    def search_with_reranking(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_type: str = "documents",
        use_hybrid: bool = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform search with optional hybrid search and re-ranking.

        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            search_type: Type of search - "documents" or "code_examples"
            use_hybrid: Override hybrid search setting (None uses environment setting)

        Returns:
            List of search results, optionally re-ranked
        """
        # Determine if we should use hybrid search
        should_use_hybrid = (
            use_hybrid if use_hybrid is not None else self.use_hybrid_search
        )

        # Perform the search
        if should_use_hybrid:
            results = self.hybrid_search(
                query, match_count, filter_metadata, search_type
            )
        else:
            if search_type == "code_examples":
                results = self.search_code_examples(query, match_count, filter_metadata)
            else:
                results = self.search_documents(query, match_count, filter_metadata)

        # Apply re-ranking if enabled and model is available
        if self.use_reranking and self.reranking_model and results:
            logger.debug("Applying reranking to search results")
            results = self.rerank_results(query, results)

        return results

    def fuse_search_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Fuse dense and sparse search results using Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            k: RRF parameter (default: 60)

        Returns:
            List of fused results sorted by RRF score
        """
        # Create a dictionary to store combined scores
        fused_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result.get("id", str(result))
            fused_scores[doc_id] = {
                "result": result,
                "rrf_score": 1.0 / (k + rank),
                "dense_rank": rank,
                "sparse_rank": None,
            }

        # Process sparse results and add to existing or create new entries
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result.get("id", str(result))
            if doc_id in fused_scores:
                # Add sparse score to existing entry
                fused_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)
                fused_scores[doc_id]["sparse_rank"] = rank
            else:
                # Create new entry for sparse-only result
                fused_scores[doc_id] = {
                    "result": result,
                    "rrf_score": 1.0 / (k + rank),
                    "dense_rank": None,
                    "sparse_rank": rank,
                }

        # Sort by RRF score and extract results
        sorted_items = sorted(
            fused_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )

        # Add RRF metadata to results
        fused_results = []
        for doc_id, score_info in sorted_items:
            result = score_info["result"].copy()
            result["rrf_score"] = score_info["rrf_score"]
            result["dense_rank"] = score_info["dense_rank"]
            result["sparse_rank"] = score_info["sparse_rank"]
            fused_results.append(result)

        return fused_results


def add_documents_to_vector_db(
    client: QdrantClientWrapper,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 500,
    file_ids: Optional[List[str]] = None,
) -> None:
    """
    Add documents to Qdrant crawled_pages collection.

    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
        file_ids: Optional list of file_id strings for cross-system linking with Neo4j
    """
    # Import required functions and modules
    import concurrent.futures
    from qdrant_client.models import PointStruct

    try:
        from ..services.embedding_service import (
            create_embeddings_batch,
            process_chunk_with_context,
        )
    except ImportError:
        from services.embedding_service import (
            create_embeddings_batch,
            process_chunk_with_context,
        )

    if not urls:
        return

    # Check if contextual embeddings are enabled
    use_contextual_embeddings = (
        os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    )
    logger.info(f"Use contextual embeddings: {use_contextual_embeddings}")

    # Add file_id to metadatas if provided
    enhanced_metadatas = []
    for i, metadata in enumerate(metadatas):
        enhanced_metadata = metadata.copy()
        if file_ids and i < len(file_ids) and file_ids[i]:
            enhanced_metadata["file_id"] = file_ids[i]
            logger.debug(f"Added file_id '{file_ids[i]}' to metadata for document {i}")
        enhanced_metadatas.append(enhanced_metadata)

    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(
        client.add_documents_to_qdrant(
            urls,
            chunk_numbers,
            contents,
            enhanced_metadatas,
            url_to_full_document,
            batch_size,
        )
    )

    # Process each batch
    for batch_idx, points_batch in enumerate(point_batches):
        batch_contents = [point["content"] for point in points_batch]

        # Apply contextual embedding if enabled
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for i, point in enumerate(points_batch):
                url = point["payload"]["url"]
                content = point["content"]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): idx
                    for idx, arg in enumerate(process_args)
                }

                # Process results as they complete
                results = [None] * len(process_args)  # Pre-allocate to maintain order
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        results[idx] = result
                        if success:
                            points_batch[idx]["payload"]["contextual_embedding"] = True
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        results[idx] = batch_contents[idx]

                contextual_contents = results
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the batch (supports both dense and sparse vectors)
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if use_hybrid_search:
            batch_embeddings, batch_sparse_vectors = create_embeddings_batch(
                contextual_contents
            )
            logger.info(
                f"Created {len(batch_embeddings)} dense and {len(batch_sparse_vectors)} sparse vectors for batch"
            )
        else:
            batch_embeddings = create_embeddings_batch(contextual_contents)
            batch_sparse_vectors = None
            logger.info(f"Created {len(batch_embeddings)} dense vectors for batch")

        # Create PointStruct objects with appropriate vector configuration
        qdrant_points = []
        for i, point in enumerate(points_batch):
            if use_hybrid_search:
                # Create PointStruct with named vectors (dense + sparse)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector={
                            "text-dense": batch_embeddings[i],
                            "text-sparse": batch_sparse_vectors[
                                i
                            ].to_qdrant_sparse_vector(),
                        },
                        payload=point["payload"],
                    )
                )
            else:
                # Create PointStruct with single vector (legacy mode)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=batch_embeddings[i],
                        payload=point["payload"],
                    )
                )

        # Upsert batch to Qdrant
        try:
            client.upsert_points("crawled_pages", qdrant_points)
            logger.info(
                f"Successfully inserted batch {batch_idx + 1}/{len(point_batches)}"
            )
        except Exception as e:
            logger.error(f"Error inserting batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("crawled_pages", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    logger.error(
                        f"Failed to insert individual point {point.id}: {individual_error}"
                    )

            if successful_inserts > 0:
                logger.info(
                    f"Successfully inserted {successful_inserts}/{len(qdrant_points)} points individually"
                )


def add_code_examples_to_vector_db(
    client: QdrantClientWrapper,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 100,
    file_ids: Optional[List[str]] = None,
):
    """
    Add code examples to Qdrant code_examples collection.

    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
        file_ids: Optional list of file_id strings for cross-system linking with Neo4j
    """
    # Import required functions and modules
    from qdrant_client.models import PointStruct

    try:
        from ..services.embedding_service import (
            create_embeddings_batch,
            create_embedding,
            create_sparse_embedding,
        )
        from ..sparse_vector_types import SparseVectorConfig
    except ImportError:
        from services.embedding_service import (
            create_embeddings_batch,
            create_embedding,
            create_sparse_embedding,
        )
        from sparse_vector_types import SparseVectorConfig

    if not urls:
        return

    # Add file_id to metadatas if provided
    enhanced_metadatas = []
    for i, metadata in enumerate(metadatas):
        enhanced_metadata = metadata.copy()
        if file_ids and i < len(file_ids) and file_ids[i]:
            enhanced_metadata["file_id"] = file_ids[i]
            logger.debug(
                f"Added file_id '{file_ids[i]}' to metadata for code example {i}"
            )
        enhanced_metadatas.append(enhanced_metadata)

    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(
        client.add_code_examples_to_qdrant(
            urls,
            chunk_numbers,
            code_examples,
            summaries,
            enhanced_metadatas,
            batch_size,
        )
    )

    # Process each batch
    for batch_idx, points_batch in enumerate(point_batches):
        # Create embeddings for combined text (code + summary)
        combined_texts = [point["combined_text"] for point in points_batch]

        # Support both dense and sparse vectors for hybrid search
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"

        if use_hybrid_search:
            embeddings, sparse_vectors = create_embeddings_batch(combined_texts)
            logger.info(
                f"Created {len(embeddings)} dense and {len(sparse_vectors)} sparse vectors for code examples batch"
            )
        else:
            embeddings = create_embeddings_batch(combined_texts)
            sparse_vectors = None
            logger.info(
                f"Created {len(embeddings)} dense vectors for code examples batch"
            )

        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        valid_sparse_vectors = [] if use_hybrid_search else None

        for i, embedding in enumerate(embeddings):
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
                if use_hybrid_search:
                    valid_sparse_vectors.append(sparse_vectors[i])
            else:
                logger.warning(
                    "Zero or invalid embedding detected, creating new one..."
                )
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(combined_texts[i])
                valid_embeddings.append(single_embedding)

                if use_hybrid_search:
                    # Create fallback sparse vector
                    try:
                        fallback_sparse = create_sparse_embedding(combined_texts[i])
                        valid_sparse_vectors.append(fallback_sparse)
                    except Exception as e:
                        logger.error(f"Failed to create fallback sparse vector: {e}")
                        valid_sparse_vectors.append(
                            SparseVectorConfig(indices=[], values=[])
                        )

        # Create PointStruct objects with appropriate vector configuration
        qdrant_points = []
        for i, point in enumerate(points_batch):
            if use_hybrid_search:
                # Create PointStruct with named vectors (dense + sparse)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector={
                            "text-dense": valid_embeddings[i],
                            "text-sparse": valid_sparse_vectors[
                                i
                            ].to_qdrant_sparse_vector(),
                        },
                        payload=point["payload"],
                    )
                )
            else:
                # Create PointStruct with single vector (legacy mode)
                qdrant_points.append(
                    PointStruct(
                        id=point["id"],
                        vector=valid_embeddings[i],
                        payload=point["payload"],
                    )
                )

        # Upsert batch to Qdrant
        try:
            client.upsert_points("code_examples", qdrant_points)
            logger.info(
                f"Inserted batch {batch_idx + 1} of {len(point_batches)} code examples"
            )
        except Exception as e:
            logger.error(f"Error inserting code examples batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("code_examples", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    logger.error(
                        f"Failed to insert individual code example {point.id}: {individual_error}"
                    )

            if successful_inserts > 0:
                logger.info(
                    f"Successfully inserted {successful_inserts}/{len(qdrant_points)} code examples individually"
                )


# Standalone functions for backward compatibility
def search_documents(
    qdrant_client,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    reranker=None,
) -> List[Dict[str, Any]]:
    """
    Standalone function for searching documents (used by MCP tools).

    Args:
        qdrant_client: Qdrant client instance
        query: Query text
        source: Optional source domain to filter results
        match_count: Maximum number of results to return
        reranker: Optional reranking model

    Returns:
        List of matching documents
    """

    # Create a temporary RAG service instance
    rag_service = RagService(qdrant_client, reranking_model=reranker)

    # Build filter metadata from source
    filter_metadata = None
    if source:
        filter_metadata = {"source": source}

    return rag_service.search_with_reranking(
        query, match_count, filter_metadata, search_type="documents"
    )


def search_code_examples(
    qdrant_client,
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5,
    reranker=None,
) -> List[Dict[str, Any]]:
    """
    Standalone function for searching code examples (used by MCP tools).

    Args:
        qdrant_client: Qdrant client instance
        query: Query text
        source_id: Optional source ID to filter results
        match_count: Maximum number of results to return
        reranker: Optional reranking model

    Returns:
        List of matching code examples
    """

    # Create a temporary RAG service instance
    rag_service = RagService(qdrant_client, reranking_model=reranker)

    return rag_service.search_with_reranking(
        query,
        match_count,
        filter_metadata={"source": source_id} if source_id else None,
        search_type="code_examples",
    )


def update_source_info(qdrant_client, source_id: str, summary: str, word_count: int):
    """
    Standalone function for updating source info (used by MCP tools).

    Args:
        qdrant_client: Qdrant client instance
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """

    # Create a temporary RAG service instance
    rag_service = RagService(qdrant_client)

    return rag_service.update_source_info(source_id, summary, word_count)
