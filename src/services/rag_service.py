"""
RAG (Retrieval Augmented Generation) service.

This module provides services for RAG operations including document search,
code example search, and result reranking.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from ..clients.embedding_client import EmbeddingClient
from ..clients.supabase_client import SupabaseService
from ..config import config


class RAGService:
    """Service for RAG operations."""
    
    def __init__(self, supabase_service: SupabaseService, reranking_model: Optional[CrossEncoder] = None):
        self.supabase_service = supabase_service
        self.embedding_client = EmbeddingClient()
        self.reranking_model = reranking_model
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results
            content_key: The key in each result dict that contains the text content
            
        Returns:
            Reranked list of results
        """
        if not self.reranking_model or not results:
            return results
        
        try:
            # Extract content from results
            texts = [result.get(content_key, "") for result in results]
            
            # Create pairs of [query, document] for the cross-encoder
            pairs = [[query, text] for text in texts]
            
            # Get relevance scores from the cross-encoder
            scores = self.reranking_model.predict(pairs)
            
            # Add scores to results and sort by score (descending)
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked
        except Exception as e:
            print(f"Error during reranking: {e}")
            return results
    
    def search_documents(
        self,
        query: str,
        source: Optional[str] = None,
        match_count: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a document search with optional hybrid search and reranking.
        
        Args:
            query: The search query
            source: Optional source domain to filter results
            match_count: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if match_count is None:
            match_count = config.DEFAULT_MATCH_COUNT
        
        # Check if hybrid search is enabled
        use_hybrid_search = config.USE_HYBRID_SEARCH
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}
        
        # Create query embedding
        query_embedding = self.embedding_client.create_embedding(query)
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            multiplier = config.HYBRID_SEARCH_MULTIPLIER
            vector_results = self.supabase_service.search_documents(
                query_embedding=query_embedding,
                match_count=match_count * multiplier,  # Get more to have room for filtering
                filter_metadata=filter_metadata
            )
            
            # 2. Get keyword search results using ILIKE
            client = self.supabase_service.get_client()
            crawled_pages_table = config.TABLE_CRAWLED_PAGES
            keyword_query = client.from_(crawled_pages_table)\
                .select('id, url, chunk_number, content, metadata, source_id')\
                .ilike('content', f'%{query}%')
            
            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq('source_id', source)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * multiplier).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = self.supabase_service.search_documents(
                query_embedding=query_embedding,
                match_count=match_count,
                filter_metadata=filter_metadata
            )
        
        # Apply reranking if enabled
        use_reranking = config.USE_RERANKING
        if use_reranking and self.reranking_model:
            results = self.rerank_results(query, results, content_key="content")
        
        return results
    
    def search_code_examples(
        self,
        query: str,
        source_id: Optional[str] = None,
        match_count: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples with optional hybrid search and reranking.
        
        Args:
            query: The search query
            source_id: Optional source ID to filter results
            match_count: Maximum number of results to return
            
        Returns:
            List of code example search results
        """
        if match_count is None:
            match_count = config.DEFAULT_MATCH_COUNT
        
        # Check if code example extraction is enabled
        extract_code_examples_enabled = config.USE_AGENTIC_RAG
        if not extract_code_examples_enabled:
            raise ValueError("Code example extraction is disabled. Perform a normal RAG search.")
        
        # Check if hybrid search is enabled
        use_hybrid_search = config.USE_HYBRID_SEARCH
        
        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}
        
        # Create a more descriptive query for better embedding match
        enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
        query_embedding = self.embedding_client.create_embedding(enhanced_query)
        
        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            multiplier = config.HYBRID_SEARCH_MULTIPLIER
            vector_results = self.supabase_service.search_code_examples(
                query_embedding=query_embedding,
                match_count=match_count * multiplier,  # Get more to have room for filtering
                filter_metadata=filter_metadata,
                source_id=source_id
            )
            
            # 2. Get keyword search results using ILIKE on both content and summary
            client = self.supabase_service.get_client()
            code_examples_table = config.TABLE_CODE_EXAMPLES
            keyword_query = client.from_(code_examples_table)\
                .select('id, url, chunk_number, content, summary, metadata, source_id')\
                .or_(f'content.ilike.%{query}%,summary.ilike.%{query}%')
            
            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq('source_id', source_id)
            
            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * multiplier).execute()
            keyword_results = keyword_response.data if keyword_response.data else []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append({
                        'id': kr['id'],
                        'url': kr['url'],
                        'chunk_number': kr['chunk_number'],
                        'content': kr['content'],
                        'summary': kr['summary'],
                        'metadata': kr['metadata'],
                        'source_id': kr['source_id'],
                        'similarity': 0.5  # Default similarity for keyword-only matches
                    })
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            # Standard vector search only
            results = self.supabase_service.search_code_examples(
                query_embedding=query_embedding,
                match_count=match_count,
                filter_metadata=filter_metadata,
                source_id=source_id
            )
        
        # Apply reranking if enabled
        use_reranking = config.USE_RERANKING
        if use_reranking and self.reranking_model:
            results = self.rerank_results(query, results, content_key="content")
        
        return results