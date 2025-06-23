"""
Crawling tools for MCP server.

This module contains MCP tools for web crawling operations:
- crawl_single_page: Crawl a single web page
- smart_crawl_url: Intelligently crawl based on URL type
"""

import json
import asyncio
import concurrent.futures
from urllib.parse import urlparse
from typing import Optional

from mcp.server.fastmcp import Context

from ..services.crawling import CrawlingService
from ..services.content_processing import ContentProcessingService
from ..clients.supabase_client import SupabaseService
from ..utils.content_utils import smart_chunk_markdown, extract_section_info, extract_code_blocks
from ..config import config


def register_crawl_tools(mcp):
    """Register crawling tools with the MCP server."""
    
    @mcp.tool()
    async def crawl_single_page(ctx: Context, url: str) -> str:
        """
        Crawl a single web page and store its content in Supabase.
        
        This tool is ideal for quickly retrieving content from a specific URL without following links.
        The content is stored in Supabase for later retrieval and querying.
        
        Args:
            ctx: The MCP server provided context
            url: URL of the web page to crawl
        
        Returns:
            Summary of the crawling operation and storage in Supabase
        """
        try:
            # Get services from the context
            crawler = ctx.request_context.lifespan_context.crawler
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            
            # Initialize services
            crawling_service = CrawlingService(crawler)
            content_service = ContentProcessingService()
            supabase_service = SupabaseService()
            supabase_service.client = supabase_client
            
            # Crawl the page
            crawl_result = await crawling_service.crawl_single_page(url)
            
            if crawl_result['success'] and crawl_result['markdown']:
                # Extract source_id
                parsed_url = urlparse(url)
                source_id = parsed_url.netloc or parsed_url.path
                
                # Chunk the content
                chunks = smart_chunk_markdown(crawl_result['markdown'])
                
                # Prepare data for Supabase
                urls = []
                chunk_numbers = []
                contents = []
                metadatas = []
                total_word_count = 0
                
                for i, chunk in enumerate(chunks):
                    urls.append(url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = url
                    meta["source"] = source_id
                    meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)
                    
                    # Accumulate word count
                    total_word_count += meta.get("word_count", 0)
                
                # Create url_to_full_document mapping
                url_to_full_document = {url: crawl_result['markdown']}
                
                # Update source information FIRST (before inserting documents)
                source_summary = content_service.extract_source_summary(source_id, crawl_result['markdown'][:5000])
                supabase_service.update_source_info(source_id, source_summary, total_word_count)
                
                # Add documentation chunks to Supabase (AFTER source exists)
                supabase_service.add_documents_to_supabase(urls, chunk_numbers, contents, metadatas, url_to_full_document)
                
                # Extract and process code examples only if enabled
                code_blocks = []
                if config.USE_AGENTIC_RAG:
                    code_blocks = extract_code_blocks(crawl_result['markdown'])
                    if code_blocks:
                        code_urls = []
                        code_chunk_numbers = []
                        code_examples = []
                        code_summaries = []
                        code_metadatas = []
                        
                        # Process code examples in parallel
                        summaries = content_service.process_code_examples_parallel(code_blocks)
                        
                        # Prepare code example data
                        for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                            code_urls.append(url)
                            code_chunk_numbers.append(i)
                            code_examples.append(block['code'])
                            code_summaries.append(summary)
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": i,
                                "url": url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split())
                            }
                            code_metadatas.append(code_meta)
                        
                        # Add code examples to Supabase
                        supabase_service.add_code_examples_to_supabase(
                            code_urls, code_chunk_numbers, code_examples, code_summaries, code_metadatas
                        )
                
                return json.dumps({
                    "success": True,
                    "url": url,
                    "chunks_stored": len(chunks),
                    "code_examples_stored": len(code_blocks),
                    "content_length": len(crawl_result['markdown']),
                    "total_word_count": total_word_count,
                    "source_id": source_id,
                    "links_count": {
                        "internal": len(crawl_result['links'].get("internal", [])) if crawl_result['links'] else 0,
                        "external": len(crawl_result['links'].get("external", [])) if crawl_result['links'] else 0
                    }
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": crawl_result['error_message']
                }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)

    @mcp.tool()
    async def smart_crawl_url(ctx: Context, url: str, max_depth: Optional[int] = None, max_concurrent: Optional[int] = None, chunk_size: Optional[int] = None) -> str:
        """
        Intelligently crawl a URL based on its type and store content in Supabase.
        
        This tool automatically detects the URL type and applies the appropriate crawling method:
        - For sitemaps: Extracts and crawls all URLs in parallel
        - For text files (llms.txt): Directly retrieves the content
        - For regular webpages: Recursively crawls internal links up to the specified depth
        
        All crawled content is chunked and stored in Supabase for later retrieval and querying.
        
        Args:
            ctx: The MCP server provided context
            url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
            max_depth: Maximum recursion depth for regular URLs (default: 3)
            max_concurrent: Maximum number of concurrent browser sessions (default: 10)
            chunk_size: Maximum size of each content chunk in characters (default: 1000)
        
        Returns:
            JSON string with crawl summary and storage information
        """
        try:
            # Get services from the context
            crawler = ctx.request_context.lifespan_context.crawler
            supabase_client = ctx.request_context.lifespan_context.supabase_client
            
            # Initialize services
            crawling_service = CrawlingService(crawler)
            content_service = ContentProcessingService()
            supabase_service = SupabaseService()
            supabase_service.client = supabase_client
            
            # Set parameters from arguments or environment variables
            max_depth = max_depth if max_depth is not None else config.MAX_CRAWL_DEPTH
            max_concurrent = max_concurrent if max_concurrent is not None else config.MAX_CONCURRENT_CRAWLS
            chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
            
            # Smart crawl based on URL type
            crawl_results, crawl_type = await crawling_service.smart_crawl_url(url, max_depth, max_concurrent)
            
            if not crawl_results:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No content found"
                }, indent=2)
            
            # Process results and store in Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            chunk_count = 0
            
            # Track sources and their content
            source_content_map = {}
            source_word_counts = {}
            
            # Process documentation chunks
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
                
                # Extract source_id
                parsed_url = urlparse(source_url)
                source_id = parsed_url.netloc or parsed_url.path
                
                # Store content for source summary generation
                if source_id not in source_content_map:
                    source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                    source_word_counts[source_id] = 0
                
                for i, chunk in enumerate(chunks):
                    urls.append(source_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # Extract metadata
                    meta = extract_section_info(chunk)
                    meta["chunk_index"] = i
                    meta["url"] = source_url
                    meta["source"] = source_id
                    meta["crawl_type"] = crawl_type
                    meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                    metadatas.append(meta)
                    
                    # Accumulate word count
                    source_word_counts[source_id] += meta.get("word_count", 0)
                    
                    chunk_count += 1
            
            # Create url_to_full_document mapping
            url_to_full_document = {}
            for doc in crawl_results:
                url_to_full_document[doc['url']] = doc['markdown']
            
            # Update source information for each unique source FIRST (before inserting documents)
            source_summaries = content_service.process_source_summaries_parallel(source_content_map)
            
            for source_id, summary in source_summaries.items():
                word_count = source_word_counts.get(source_id, 0)
                supabase_service.update_source_info(source_id, summary, word_count)
            
            # Add documentation chunks to Supabase (AFTER sources exist)
            supabase_service.add_documents_to_supabase(urls, chunk_numbers, contents, metadatas, url_to_full_document)
            
            # Extract and process code examples from all documents only if enabled
            code_examples = []
            if config.USE_AGENTIC_RAG:
                all_code_blocks = []
                code_urls = []
                code_chunk_numbers = []
                code_examples_list = []
                code_summaries = []
                code_metadatas = []
                
                # Extract code blocks from all documents
                for doc in crawl_results:
                    source_url = doc['url']
                    md = doc['markdown']
                    code_blocks = extract_code_blocks(md)
                    
                    if code_blocks:
                        # Process code examples in parallel
                        summaries = content_service.process_code_examples_parallel(code_blocks)
                        
                        # Prepare code example data
                        parsed_url = urlparse(source_url)
                        source_id = parsed_url.netloc or parsed_url.path
                        
                        for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                            code_urls.append(source_url)
                            code_chunk_numbers.append(len(code_examples_list))  # Use global code example index
                            code_examples_list.append(block['code'])
                            code_summaries.append(summary)
                            
                            # Create metadata for code example
                            code_meta = {
                                "chunk_index": len(code_examples_list) - 1,
                                "url": source_url,
                                "source": source_id,
                                "char_count": len(block['code']),
                                "word_count": len(block['code'].split())
                            }
                            code_metadatas.append(code_meta)
                
                # Add all code examples to Supabase
                if code_examples_list:
                    supabase_service.add_code_examples_to_supabase(
                        code_urls, code_chunk_numbers, code_examples_list, code_summaries, code_metadatas
                    )
                    code_examples = code_examples_list
            
            return json.dumps({
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "code_examples_stored": len(code_examples),
                "sources_updated": len(source_content_map),
                "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else [])
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "error": str(e)
            }, indent=2)