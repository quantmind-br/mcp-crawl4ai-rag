"""Centralized text chunking utilities for content processing.

This module provides intelligent text chunking functionality that respects
code blocks, paragraphs, and sentence boundaries for optimal content processing
across web crawling and GitHub repository indexing.
"""

from typing import List


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Split text into chunks, respecting code blocks and paragraphs.

    This function intelligently chunks text by preferring to break at:
    1. Code block boundaries (```)
    2. Paragraph breaks (\n\n)
    3. Sentence boundaries (. )

    Args:
        text: Input text to be chunked
        chunk_size: Maximum size of each chunk in characters (default: 5000)

    Returns:
        List of text chunks, each respecting natural boundaries where possible

    Example:
        >>> chunks = smart_chunk_markdown("# Title\n\nParagraph 1.\n\nParagraph 2.", 50)
        >>> len(chunks) >= 1
        True
    """
    # Return empty list for empty or whitespace-only strings
    if not text or not text.strip():
        return []

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

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if (
                last_period > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks
