"""
Search tools for MCP server.

This module contains MCP tools for search operations:
- perform_rag_query: Perform RAG search on stored documents
- search_code_examples: Search for code examples
"""

import json
from typing import Optional

from mcp.server.fastmcp import Context

from ..services.rag_service import RAGService
from ..clients.supabase_client import SupabaseService
from ..config import config


def register_search_tools(mcp):
    """Register search tools with the MCP server."""
    
    @mcp.tool()
    async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: Optional[int] = None) -> str:
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
            # Get services from the context
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            reranking_model = ctx.request_context.lifespan_context.reranking_model
            
            # Initialize services
            supabase_service = SupabaseService()
            supabase_service.client = supabase_client
            rag_service = RAGService(supabase_service, reranking_model)
            
            # Perform search
            results = rag_service.search_documents(query, source, match_count)
            
            # Format the results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata"),
                    "similarity": result.get("similarity")
                }
                # Include rerank score if available
                if "rerank_score" in result:
                    formatted_result["rerank_score"] = result["rerank_score"]
                formatted_results.append(formatted_result)
            
            return json.dumps({
                "success": True,
                "query": query,
                "source_filter": source,
                "search_mode": "hybrid" if config.USE_HYBRID_SEARCH else "vector",
                "reranking_applied": config.USE_RERANKING and reranking_model is not None,
                "results": formatted_results,
                "count": len(formatted_results)
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "query": query,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def search_code_examples(ctx: Context, query: str, source_id: str = None, match_count: Optional[int] = None) -> str:
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
            # Get services from the context
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            reranking_model = ctx.request_context.lifespan_context.reranking_model
            
            # Initialize services
            supabase_service = SupabaseService()
            supabase_service.client = supabase_client
            rag_service = RAGService(supabase_service, reranking_model)
            
            # Perform search
            results = rag_service.search_code_examples(query, source_id, match_count)
            
            # Format the results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "url": result.get("url"),
                    "code": result.get("content"),
                    "summary": result.get("summary"),
                    "metadata": result.get("metadata"),
                    "source_id": result.get("source_id"),
                    "similarity": result.get("similarity")
                }
                # Include rerank score if available
                if "rerank_score" in result:
                    formatted_result["rerank_score"] = result["rerank_score"]
                formatted_results.append(formatted_result)
            
            return json.dumps({
                "success": True,
                "query": query,
                "source_filter": source_id,
                "search_mode": "hybrid" if config.USE_HYBRID_SEARCH else "vector",
                "reranking_applied": config.USE_RERANKING and reranking_model is not None,
                "results": formatted_results,
                "count": len(formatted_results)
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "query": query,
                "error": str(e)
            }, indent=2)