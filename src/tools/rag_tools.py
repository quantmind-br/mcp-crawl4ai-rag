"""
RAG (Retrieval-Augmented Generation) tools for MCP server.

This module contains tools for querying the vector database and retrieving
stored content using semantic search.
"""

import json
from typing import Optional

from mcp.server.fastmcp import Context

# Import utilities - these will be refactored when services are fully implemented
try:
    from ..services.rag_service import (
        search_documents,
        search_code_examples as search_code_examples_impl,
        update_source_info,
    )
except ImportError:
    from services.rag_service import (
        search_documents,
        search_code_examples as search_code_examples_impl,
    )


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
    ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get clients from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        reranker = getattr(ctx.request_context.lifespan_context, 'reranker', None)

        # Perform the search using the utility function
        results = search_documents(
            qdrant_client, query, source, match_count, reranker
        )

        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


async def search_code_examples(
    ctx: Context, query: str, source_id: Optional[str] = None, match_count: int = 5
) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get clients from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        reranker = getattr(ctx.request_context.lifespan_context, 'reranker', None)

        # Perform the search using the utility function
        results = search_code_examples_impl(
            qdrant_client, query, source_id, match_count, reranker
        )

        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)