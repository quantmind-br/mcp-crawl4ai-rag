"""
Web crawling tools for MCP server.

This module contains tools for crawling web pages, sitemaps, and text files
using Crawl4AI, with intelligent content extraction and storage.
"""

import asyncio
import concurrent.futures
import json
import os
import re
from typing import Dict, List, Any
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
import requests
import logging

from crawl4ai import (
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
    AsyncWebCrawler,
)
from mcp.server.fastmcp import Context

# Import MCP decorator - will be applied when tools are registered
# Will get actual mcp instance from parent module
mcp = None

# Import utilities from the services layer
try:
    from ..services.rag_service import (
        add_documents_to_vector_db,
        add_code_examples_to_vector_db,
        update_source_info,
    )
except ImportError:
    from services.rag_service import (
        add_documents_to_vector_db,
        add_code_examples_to_vector_db,
        update_source_info,
    )

# Import chunking utility from centralized utils module
try:
    from ..utils.chunking import smart_chunk_markdown
except ImportError:
    from utils.chunking import smart_chunk_markdown

logger = logging.getLogger(__name__)


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content.

    Args:
        source_id: The source ID (domain)
        content: The content to summarize
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    if not content:
        return f"Empty source: {source_id}"

    # Simple truncation-based summary
    lines = content.strip().split("\n")
    summary_lines = []
    total_length = 0

    for line in lines:
        line = line.strip()
        if line:
            if total_length + len(line) <= max_length - 3:  # Reserve space for "..."
                summary_lines.append(line)
                total_length += len(line) + 1  # +1 for newline
            else:
                # If this line is too long, truncate it and add it
                if not summary_lines:  # Only if we haven't added anything yet
                    truncated_line = line[: max_length - 3]
                    summary_lines.append(truncated_line)
                    total_length = len(truncated_line)
                break

    summary = " ".join(summary_lines)
    if len(content.strip()) > max_length or total_length >= max_length - 3:
        if len(summary) > max_length - 3:
            summary = summary[: max_length - 3]
        summary += "..."

    return summary or f"Content from {source_id}"


def extract_code_blocks(
    markdown_content: str, min_length: int = 1000
) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content with context.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract

    Returns:
        List of dictionaries containing code blocks with context
    """
    import re

    code_blocks = []

    # Pattern to find code blocks with context
    # Allow optional leading spaces before ``` and optional trailing spaces
    pattern = r"(?m)^[ \t]*```(\w*)\s*\n([\s\S]*?)\n[ \t]*```"

    # Split content into lines for context extraction
    lines = markdown_content.split("\n")

    # Find all matches with their positions
    for match in re.finditer(pattern, markdown_content, re.DOTALL):
        language, code = match.groups()

        # Always capture code blocks, but enforce min_length only for storage cases
        # so tests can lower the threshold and still get blocks.
        start_pos = match.start()
        end_pos = match.end()

        # Count lines before the match
        lines_before_start = markdown_content[:start_pos].count("\n")
        lines_after_end = markdown_content[:end_pos].count("\n")

        # Extract context (3 lines before and after)
        context_before_lines = max(0, lines_before_start - 3)
        context_after_lines = min(len(lines), lines_after_end + 3)

        context_before = "\n".join(lines[context_before_lines:lines_before_start])
        context_after = "\n".join(lines[lines_after_end:context_after_lines])

        # Ensure nearest preceding header (e.g., Markdown title) is included in context_before
        # Search upwards from lines_before_start for a line starting with '#'
        header_line = None
        for i in range(lines_before_start - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("#"):
                header_line = line
                break
        if header_line and header_line not in context_before:
            context_before = (header_line + "\n" + context_before).strip()

        block = {
            "code": code.strip(),
            "language": language or "text",
            "context_before": context_before.strip(),
            "context_after": context_after.strip(),
            "index": len(code_blocks),
            "length": len(code),
        }

        # Apply min_length filter at the end to allow tests to set smaller thresholds
        if len(block["code"]) >= min_length:
            code_blocks.append(block)

    return code_blocks


def generate_code_example_summary(
    code: str, context_before: str = "", context_after: str = ""
) -> str:
    """
    Generate a summary for a code example with context.

    Args:
        code: The code to summarize
        context_before: Context before the code block
        context_after: Context after the code block

    Returns:
        A summary string
    """
    lines = code.strip().split("\n")

    # Try to extract language from context or code patterns
    language = ""

    # Check for language indicators in context
    context = (context_before + " " + context_after).lower()
    if "python" in context or code.strip().startswith(
        ("def ", "class ", "import ", "from ")
    ):
        language = "Python"
    elif (
        "javascript" in context
        or "js" in context
        or code.strip().startswith(("function", "const ", "let ", "var "))
    ):
        language = "JavaScript"
    elif (
        "bash" in context
        or "shell" in context
        or code.strip().startswith(("#!/bin/bash", "$", "sudo"))
    ):
        language = "Bash"
    elif "yaml" in context or code.strip().startswith(("-", "---")):
        language = "YAML"
    elif "json" in context or (
        code.strip().startswith("{") and code.strip().endswith("}")
    ):
        language = "JSON"

    # Try to find function/class definitions
    for line in lines:
        line = line.strip()
        if line.startswith(("def ", "class ")):
            return f"{language} code: {line}"
        elif line.startswith(("function ", "const ", "let ", "var ")):
            return f"{language} code: {line}"
        elif line.startswith("#!/"):
            return f"{language} script: {line}"

    # Use context to provide better summary
    if context_before:
        context_summary = context_before.strip().split("\n")[-1][:50]
        if context_summary:
            return f"{language} code example: {context_summary}..."

    # Fallback to first non-empty line
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            truncated = line[:100] + "..." if len(line) > 100 else line
            return f"{language} code: {truncated}"

    return f"{language} code example ({len(lines)} lines)"


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith(".txt")


def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    urls = []
    try:
        resp = requests.get(sitemap_url)
        if resp.status_code == 200:
            try:
                tree = ElementTree.fromstring(resp.content)
                urls = [loc.text for loc in tree.findall(".//{*}loc")]
            except Exception as e:
                logger.warning(f"Error parsing sitemap XML: {e}")
        else:
            logger.warning(
                f"Sitemap request returned status {resp.status_code} for {sitemap_url}"
            )
    except Exception as e:
        logger.warning(f"Error fetching sitemap '{sitemap_url}': {e}")

    return urls


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (code, context_before, context_after)

    Returns:
        The generated summary
    """
    code, context_before, context_after = args
    return generate_code_example_summary(code, context_before, context_after)


async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    else:
        logger.warning(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    results = await crawler.arun_many(
        urls=urls, config=crawl_config, dispatcher=dispatcher
    )
    return [
        {"url": r.url, "markdown": r.markdown}
        for r in results
        if r.success and r.markdown
    ]


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    base_prefix: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        base_prefix: URL prefix to restrict crawling scope (only URLs starting with this prefix will be crawled)
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(
            urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
        )
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({"url": result.url, "markdown": result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited and next_url.startswith(base_prefix):
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in the vector database.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in the vector database for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in the vector database
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path

            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for vector database
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
            url_to_full_document = {url: result.markdown}

            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(
                source_id, result.markdown[:5000]
            )  # Use first 5000 chars for summary
            update_source_info(
                qdrant_client, source_id, source_summary, total_word_count
            )

            # Add documentation chunks to Qdrant (AFTER source exists)
            add_documents_to_vector_db(
                qdrant_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )

            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            code_examples_count = 0
            if extract_code_examples:
                code_blocks = extract_code_blocks(result.markdown)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []

                    # Process code examples in parallel
                    # Import performance config for optimized worker count
                    try:
                        from ..utils.performance_config import get_performance_config
                    except ImportError:
                        from utils.performance_config import get_performance_config

                    config = get_performance_config()
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=config.io_workers
                    ) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [
                            (
                                block["code"],
                                block["context_before"],
                                block["context_after"],
                            )
                            for block in code_blocks
                        ]

                        # Generate summaries in parallel
                        summaries = list(
                            executor.map(process_code_example, summary_args)
                        )

                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block["code"])
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": i,
                            "url": url,
                            "source": source_id,
                            "char_count": len(block["code"]),
                            "word_count": len(block["code"].split()),
                        }
                        code_metadatas.append(code_meta)

                    # Add code examples to Qdrant
                    add_code_examples_to_vector_db(
                        qdrant_client,
                        code_urls,
                        code_chunk_numbers,
                        code_examples,
                        code_summaries,
                        code_metadatas,
                    )
                    code_examples_count = len(code_examples)

            links = getattr(result, "links", {}) or {}
            internal_links = (
                links.get("internal", []) if isinstance(links, dict) else []
            )
            external_links = (
                links.get("external", []) if isinstance(links, dict) else []
            )

            return json.dumps(
                {
                    "success": True,
                    "url": url,
                    "chunks_stored": len(chunks),
                    "code_examples_stored": code_examples_count,
                    "content_length": len(result.markdown),
                    "total_word_count": total_word_count,
                    "source_id": source_id,
                    "links_count": {
                        "internal": len(internal_links),
                        "external": len(external_links),
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"success": False, "url": url, "error": result.error_message}, indent=2
            )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


def create_base_prefix(url: str) -> str:
    """
    Create a proper base prefix from URL for crawling scope restriction.

    Examples:
    - https://opencode.ai/ -> https://opencode.ai/
    - https://opencode.ai -> https://opencode.ai/
    - https://docs.anthropic.com/en/ -> https://docs.anthropic.com/en/
    - https://docs.anthropic.com/en -> https://docs.anthropic.com/en/

    Args:
        url: Input URL

    Returns:
        Normalized base prefix for crawling scope
    """
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    path = parsed_url.path

    # Ensure we have a scheme
    if not scheme:
        scheme = "https"

    # Build base prefix
    if not path or path == "/":
        # Root level: https://example.com/ -> https://example.com/
        base_prefix = f"{scheme}://{netloc}/"
    else:
        # Path specified: ensure it ends with / for proper prefix matching
        path = path.rstrip("/") + "/"
        base_prefix = f"{scheme}://{netloc}{path}"

    return base_prefix


async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """
    Intelligently crawl a URL based on its type and store content in the vector database.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in the vector database for later retrieval and querying.

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
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client

        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None

        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps(
                    {"success": False, "url": url, "error": "No URLs found in sitemap"},
                    indent=2,
                )
            crawl_results = await crawl_batch(
                crawler, sitemap_urls, max_concurrent=max_concurrent
            )
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl with proper base prefix
            base_prefix = create_base_prefix(url)

            crawl_results = await crawl_recursive_internal_links(
                crawler,
                [url],
                base_prefix=base_prefix,
                max_depth=max_depth,
                max_concurrent=max_concurrent,
            )
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps(
                {"success": False, "url": url, "error": "No content found"}, indent=2
            )

        # Process results and store in vector database
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
            source_url = doc["url"]
            md = doc["markdown"]
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
            url_to_full_document[doc["url"]] = doc["markdown"]

        # Update source information for each unique source FIRST (before inserting documents)
        # Import performance config for optimized worker count
        try:
            from ..utils.performance_config import get_performance_config
        except ImportError:
            from utils.performance_config import get_performance_config

        config = get_performance_config()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(config.io_workers, 20)
        ) as executor:
            source_summary_args = [
                (source_id, content)
                for source_id, content in source_content_map.items()
            ]
            source_summaries = list(
                executor.map(
                    lambda args: extract_source_summary(args[0], args[1]),
                    source_summary_args,
                )
            )

        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(qdrant_client, source_id, summary, word_count)

        # Add documentation chunks to Qdrant (AFTER sources exist)
        batch_size = 20
        add_documents_to_vector_db(
            qdrant_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            batch_size=batch_size,
        )

        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []

            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc["url"]
                md = doc["markdown"]
                code_blocks = extract_code_blocks(md)

                if code_blocks:
                    # Process code examples in parallel
                    # Import performance config for optimized worker count
                    try:
                        from ..utils.performance_config import get_performance_config
                    except ImportError:
                        from utils.performance_config import get_performance_config

                    config = get_performance_config()
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=config.io_workers
                    ) as executor:
                        # Prepare arguments for parallel processing
                        summary_args = [
                            (
                                block["code"],
                                block["context_before"],
                                block["context_after"],
                            )
                            for block in code_blocks
                        ]

                        # Generate summaries in parallel
                        summaries = list(
                            executor.map(process_code_example, summary_args)
                        )

                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path

                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(source_url)
                        code_chunk_numbers.append(
                            len(code_examples)
                        )  # Use global code example index
                        code_examples.append(block["code"])
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block["code"]),
                            "word_count": len(block["code"].split()),
                        }
                        code_metadatas.append(code_meta)

            # Add all code examples to Qdrant
            if code_examples:
                add_code_examples_to_vector_db(
                    qdrant_client,
                    code_urls,
                    code_chunk_numbers,
                    code_examples,
                    code_summaries,
                    code_metadatas,
                    batch_size=batch_size,
                )

        return json.dumps(
            {
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "code_examples_stored": len(code_examples),
                "sources_updated": len(source_content_map),
                "urls_crawled": [doc["url"] for doc in crawl_results][:5]
                + (["..."] if len(crawl_results) > 5 else []),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)
