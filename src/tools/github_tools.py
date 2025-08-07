"""GitHub-specific MCP tools.

This module contains MCP tools for GitHub repository processing,
including repository crawling, analysis, and content extraction.
"""

import json
import os
import concurrent.futures
from typing import List
from urllib.parse import urlparse

from mcp.server.fastmcp import Context

# MCP decorator will be applied when module is imported
# This allows for flexible registration patterns
from ..features.github_processor import (
    GitHubRepoManager,
    GitHubMetadataExtractor,
    MultiFileDiscovery,
    PythonProcessor,
    TypeScriptProcessor,
    ConfigProcessor,
    MarkdownProcessor,
)
from ..utils.validation import validate_github_url

# Import utility functions from refactored modules
try:
    from ..services.rag_service import (
        add_documents_to_vector_db,
        add_code_examples_to_vector_db,
        update_source_info,
    )
    from ..clients.qdrant_client import get_qdrant_client
    from .web_tools import extract_source_summary
except ImportError:
    from services.rag_service import (
        add_documents_to_vector_db,
        add_code_examples_to_vector_db,
        update_source_info,
    )
    from web_tools import extract_source_summary


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if (
                last_break > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # Extract the chunk
        chunks.append(text[start:end].strip())
        start = end

    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


async def smart_crawl_github(
    ctx: Context,
    repo_url: str,
    max_files: int = 50,
    chunk_size: int = 5000,
    max_size_mb: int = 500,
    file_types_to_index: List[str] = [".md"],
) -> str:
    """
    Clone a GitHub repository, extract content from multiple file types,
    and store them in the vector database.

    This tool clones a GitHub repository to a temporary directory, discovers files
    of specified types, extracts content and metadata, and stores the content in
    chunks for vector search and retrieval.

    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo')
        max_files: Maximum number of files to process (default: 50)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
        max_size_mb: Maximum repository size in MB (default: 500)
        file_types_to_index: File extensions to process (default: ['.md'])
                           Supported: ['.md', '.py', '.ts', '.tsx', '.json', '.yaml', '.yml', '.toml']

    Returns:
        JSON string with crawl summary and storage information
    """
    repo_manager = None
    try:
        # Validate GitHub URL
        is_valid, error_msg = validate_github_url(repo_url)
        if not is_valid:
            return json.dumps(
                {
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Invalid GitHub URL: {error_msg}",
                },
                indent=2,
            )

        # Get clients from context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client

        # Initialize GitHub processing components
        repo_manager = GitHubRepoManager()
        file_discovery = MultiFileDiscovery()
        metadata_extractor = GitHubMetadataExtractor()

        # Clone the repository
        repo_path = repo_manager.clone_repository(repo_url, max_size_mb)

        # Extract repository metadata
        repo_metadata = metadata_extractor.extract_repo_metadata(repo_url, repo_path)

        # Discover files of specified types
        discovered_files = file_discovery.discover_files(
            repo_path, file_types=file_types_to_index, max_files=max_files
        )

        if not discovered_files:
            return json.dumps(
                {
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"No files found in repository for types: {file_types_to_index}",
                },
                indent=2,
            )

        # Process files by type using appropriate processors
        processor_map = {
            ".md": MarkdownProcessor,
            ".markdown": MarkdownProcessor,
            ".mdown": MarkdownProcessor,
            ".mkd": MarkdownProcessor,
            ".py": PythonProcessor,
            ".ts": TypeScriptProcessor,
            ".tsx": TypeScriptProcessor,
            ".json": ConfigProcessor,
            ".yaml": ConfigProcessor,
            ".yml": ConfigProcessor,
            ".toml": ConfigProcessor,
        }

        processed_documents = []
        total_chunks = 0
        file_type_stats = {}

        # Extract source_id from repo URL
        parsed_url = urlparse(repo_url)
        base_source_id = parsed_url.netloc + parsed_url.path.rstrip("/")

        # Store repository content for source summary generation
        repo_content = ""
        total_word_count = 0

        # Process each discovered file
        for file_info in discovered_files:
            file_path = file_info["path"]
            relative_path = file_info["relative_path"]
            file_ext = file_info["file_type"]

            # Get appropriate processor
            if file_ext in processor_map:
                processor = processor_map[file_ext]()
                extracted_items = processor.process_file(file_path, relative_path)

                # Update file type statistics
                if file_ext not in file_type_stats:
                    file_type_stats[file_ext] = {"files": 0, "items": 0}
                file_type_stats[file_ext]["files"] += 1
                file_type_stats[file_ext]["items"] += len(extracted_items)

                for item in extracted_items:
                    # Create document for chunking
                    content = item["content"]
                    file_url = f"{repo_url}/blob/main/{relative_path}"

                    # Accumulate content for repository summary
                    repo_content += content[:1000] + "\n\n"  # First 1000 chars per item
                    content_word_count = len(content.split())
                    total_word_count += content_word_count

                    # Create metadata
                    metadata = {
                        "file_path": relative_path,
                        "type": item["type"],
                        "name": item["name"],
                        "signature": item.get("signature"),
                        "line_number": item.get("line_number"),
                        "language": item["language"],
                        "repo_url": repo_url,
                        "source_type": "github_repository",
                        "url": file_url,
                        "source": base_source_id,
                        "filename": file_info["filename"],
                        "file_size_bytes": file_info["size_bytes"],
                        "is_readme": file_info["is_readme"],
                        "crawl_type": "github_repository",
                        **repo_metadata,  # Include all repository metadata
                    }

                    # Use existing chunking pipeline
                    chunks = smart_chunk_markdown(content, chunk_size)

                    for i, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update(
                            {
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "word_count": len(chunk.split()),
                                "char_count": len(chunk),
                            }
                        )

                        processed_documents.append(
                            {"content": chunk, "metadata": chunk_metadata}
                        )
                        total_chunks += 1

        # Generate and update source summary
        repo_summary = extract_source_summary(base_source_id, repo_content[:5000])
        update_source_info(qdrant_client, base_source_id, repo_summary, total_word_count)

        # Store processed documents using new format
        if processed_documents:
            # Extract data for existing storage function
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            url_to_full_document = {}

            for doc in processed_documents:
                contents.append(doc["content"])
                metadatas.append(doc["metadata"])
                urls.append(doc["metadata"]["url"])
                chunk_numbers.append(doc["metadata"]["chunk_index"])

                # Build url_to_full_document mapping (use first 5000 chars as full document)
                url = doc["metadata"]["url"]
                if url not in url_to_full_document:
                    url_to_full_document[url] = doc["content"][:5000]

            # Add documents to Qdrant using existing function
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

        # Extract and process code examples if enabled
        code_examples_count = 0
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if extract_code_examples_enabled:
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []

            # Extract code blocks from markdown files only
            markdown_files = [
                f
                for f in discovered_files
                if f["file_type"] in [".md", ".markdown", ".mdown", ".mkd"]
            ]
            for file_info in markdown_files:
                file_url = f"{repo_url}/blob/main/{file_info['relative_path']}"

                # Read markdown content for code block extraction
                try:
                    with open(
                        file_info["path"], "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        content = f.read()
                    code_blocks = extract_code_blocks(content)
                except Exception:
                    continue  # Skip if can't read file

                if code_blocks:
                    # Process code examples in parallel
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=10
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
                            executor.map(
                                lambda args: generate_code_example_summary(
                                    args[0], args[1], args[2]
                                ),
                                summary_args,
                            )
                        )

                    # Prepare code example data
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries)):
                        code_urls.append(file_url)
                        code_chunk_numbers.append(len(code_examples))
                        code_examples.append(block["code"])
                        code_summaries.append(summary)

                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": file_url,
                            "source": base_source_id,
                            "filename": file_info["filename"],
                            "relative_path": file_info["relative_path"],
                            "language": block.get("language", ""),
                            "char_count": len(block["code"]),
                            "word_count": len(block["code"].split()),
                            **repo_metadata,  # Include repository metadata
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
                code_examples_count = len(code_examples)

        return json.dumps(
            {
                "success": True,
                "repo_url": repo_url,
                "owner": repo_metadata.get("owner", ""),
                "repo_name": repo_metadata.get("repo_name", ""),
                "file_types_requested": file_types_to_index,
                "files_discovered": len(discovered_files),
                "file_type_stats": file_type_stats,
                "chunks_stored": total_chunks,
                "code_examples_stored": code_examples_count,
                "total_word_count": total_word_count,
                "repository_size_mb": repo_manager._get_directory_size_mb(repo_path),
                "source_id": base_source_id,
                "files_processed": [f["relative_path"] for f in discovered_files[:10]]
                + (["..."] if len(discovered_files) > 10 else []),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "repo_url": repo_url, "error": str(e)}, indent=2
        )
    finally:
        # Always cleanup temporary directories
        if repo_manager:
            repo_manager.cleanup()