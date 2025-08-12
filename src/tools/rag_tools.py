"""
RAG (Retrieval-Augmented Generation) tools for MCP server.

This module contains tools for querying the vector database and retrieving
stored content using semantic search.
"""

import json
from typing import Optional

from mcp.server.fastmcp import Context

# RAG service will be imported directly in functions as needed


async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get Qdrant client from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client

        # Search for all documents in the sources collection to get unique sources
        sources = qdrant_client.get_available_sources()
        return json.dumps(
            {
                "success": True,
                "sources": sources,
                "total_sources": len(sources),
                "message": "Use these source IDs to filter your RAG queries",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def perform_rag_query(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
    file_id: Optional[str] = None,
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain and/or specific file_id.
    Get the source by using the get_available_sources tool before calling this search!

    The file_id parameter enables precise filtering to documents from specific files,
    supporting cross-system linking with Neo4j knowledge graph data. File IDs follow
    the format "repo_name:relative_path" (e.g., "pydantic-ai:docs/agents.md").

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'github.com/user/repo')
        match_count: Maximum number of results to return (default: 5)
        file_id: Optional file_id to filter results to specific files (e.g., 'repo:path/file.md')

    Returns:
        JSON string with the search results
    """
    try:
        # Get clients from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        reranker = getattr(ctx.request_context.lifespan_context, "reranker", None)

        # Build filter metadata with support for both source and file_id filtering
        filter_metadata = {}
        if source:
            filter_metadata["source"] = source
        if file_id:
            filter_metadata["file_id"] = file_id

        # Use updated search function that accepts filter_metadata
        from ..services.rag_service import RagService

        rag_service = RagService(qdrant_client, reranking_model=reranker)

        results = rag_service.search_with_reranking(
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata if filter_metadata else None,
            search_type="documents",
        )

        # Format response with filtering information
        response = {
            "success": True,
            "query": query,
            "match_count": len(results),
            "filters_applied": {"source": source, "file_id": file_id},
            "results": results,
        }

        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


async def search_code_examples(
    ctx: Context,
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5,
    file_id: Optional[str] = None,
) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id and/or file_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    The file_id parameter enables precise filtering to code examples from specific files,
    supporting cross-system linking with Neo4j knowledge graph data. File IDs follow
    the format "repo_name:relative_path" (e.g., "pydantic-ai:examples/basic.py").

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'github.com/user/repo')
        match_count: Maximum number of results to return (default: 5)
        file_id: Optional file_id to filter results to specific files (e.g., 'repo:path/file.py')

    Returns:
        JSON string with the search results
    """
    try:
        # Get clients from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        reranker = getattr(ctx.request_context.lifespan_context, "reranker", None)

        # Build filter metadata with support for both source_id and file_id filtering
        filter_metadata = {}
        if source_id:
            filter_metadata["source"] = source_id
        if file_id:
            filter_metadata["file_id"] = file_id

        # Use updated search function that accepts filter_metadata
        from ..services.rag_service import RagService

        rag_service = RagService(qdrant_client, reranking_model=reranker)

        results = rag_service.search_with_reranking(
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata if filter_metadata else None,
            search_type="code_examples",
        )

        # Format response with filtering information
        response = {
            "success": True,
            "query": query,
            "match_count": len(results),
            "filters_applied": {"source_id": source_id, "file_id": file_id},
            "results": results,
        }

        return json.dumps(response, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)
