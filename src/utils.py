"""
Utility functions for the Crawl4AI MCP server with Qdrant integration.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
from urllib.parse import urlparse
import openai
import re
import time
import logging

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue, MatchText
from qdrant_client import QdrantClient

# Import our Qdrant client wrapper
try:
    from .qdrant_wrapper import QdrantClientWrapper, get_qdrant_client
except ImportError:
    from qdrant_wrapper import QdrantClientWrapper, get_qdrant_client

# Load OpenAI API key for embeddings (backward compatibility)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chat_client():
    """
    Get a configured OpenAI client for chat/completion operations.
    
    Supports flexible configuration through environment variables:
    - CHAT_API_KEY: API key for chat model (falls back to OPENAI_API_KEY)
    - CHAT_API_BASE: Base URL for chat API (defaults to OpenAI)
    
    Returns:
        openai.OpenAI: Configured OpenAI client for chat operations
        
    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration with fallback logic
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("CHAT_API_BASE")
    
    if not api_key:
        raise ValueError(
            "No API key configured for chat model. Please set CHAT_API_KEY or OPENAI_API_KEY"
        )
    
    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)

def get_embeddings_client():
    """
    Get a configured OpenAI client for embeddings operations.
    
    Supports flexible configuration through environment variables:
    - EMBEDDINGS_API_KEY: API key for embeddings (falls back to OPENAI_API_KEY)
    - EMBEDDINGS_API_BASE: Base URL for embeddings API (defaults to OpenAI)
    
    Returns:
        openai.OpenAI: Configured OpenAI client for embeddings operations
        
    Raises:
        ValueError: If no API key is configured
    """
    # Get configuration with fallback logic
    api_key = os.getenv("EMBEDDINGS_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("EMBEDDINGS_API_BASE")
    
    if not api_key:
        raise ValueError(
            "No API key configured for embeddings. Please set EMBEDDINGS_API_KEY or OPENAI_API_KEY"
        )
    
    # Create client with optional base_url
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    else:
        return openai.OpenAI(api_key=api_key)

def validate_chat_config() -> bool:
    """
    Validate chat model configuration and provide helpful guidance.
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If critical configuration is missing
    """
    # Check for API key
    chat_api_key = os.getenv("CHAT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not chat_api_key and not openai_api_key:
        raise ValueError(
            "No API key configured for chat model. Please set one of:\n"
            "  - CHAT_API_KEY (recommended for new deployments)\n"
            "  - OPENAI_API_KEY (for backward compatibility)"
        )
    
    # Check for model configuration
    chat_model = os.getenv("CHAT_MODEL")
    model_choice = os.getenv("MODEL_CHOICE")
    
    if not chat_model and not model_choice:
        logging.warning(
            "No chat model specified. Please set CHAT_MODEL environment variable. "
            "Defaulting to OpenAI's default model."
        )
    
    # Warn about deprecated usage
    if model_choice and not chat_model:
        logging.warning(
            "MODEL_CHOICE is deprecated. Please use CHAT_MODEL instead. "
            "MODEL_CHOICE support will be removed in a future version."
        )
    
    # Log configuration being used
    effective_key_source = "CHAT_API_KEY" if chat_api_key else "OPENAI_API_KEY (fallback)"
    effective_model = chat_model or model_choice or "default"
    base_url = os.getenv("CHAT_API_BASE", "default OpenAI")
    
    logging.debug(f"Chat configuration - Model: {effective_model}, Key source: {effective_key_source}, Base URL: {base_url}")
    
    return True

def validate_embeddings_config() -> bool:
    """
    Validate embeddings model configuration and provide helpful guidance.
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If critical configuration is missing
    """
    # Check for API key
    embeddings_api_key = os.getenv("EMBEDDINGS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not embeddings_api_key and not openai_api_key:
        raise ValueError(
            "No API key configured for embeddings. Please set one of:\n"
            "  - EMBEDDINGS_API_KEY (recommended for new deployments)\n"
            "  - OPENAI_API_KEY (for backward compatibility)"
        )
    
    # Log configuration being used
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    effective_key_source = "EMBEDDINGS_API_KEY" if embeddings_api_key else "OPENAI_API_KEY (fallback)"
    base_url = os.getenv("EMBEDDINGS_API_BASE", "default OpenAI")
    
    logging.debug(f"Embeddings configuration - Model: {embeddings_model}, Key source: {effective_key_source}, Base URL: {base_url}")
    
    return True

def get_supabase_client():
    """
    DEPRECATED: Legacy function name maintained for compatibility.
    Returns Qdrant client wrapper instead.
    """
    return get_qdrant_client()

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
            client = get_embeddings_client()
            response = client.embeddings.create(
                model=embeddings_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        embeddings_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
                        client = get_embeddings_client()
                        individual_response = client.embeddings.create(
                            model=embeddings_model,
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Uses the chat model configured via CHAT_MODEL environment variable (with fallback to MODEL_CHOICE).
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Get chat model with fallback to legacy MODEL_CHOICE for backward compatibility
    model_choice = os.getenv("CHAT_MODEL", os.getenv("MODEL_CHOICE"))
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        client = get_chat_client()
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

def add_documents_to_supabase(
    client: QdrantClientWrapper, 
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 100
) -> None:
    """
    Add documents to Qdrant crawled_pages collection.
    LEGACY FUNCTION NAME: Maintained for compatibility.
    
    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
    
    # Check if contextual embeddings are enabled
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(client.add_documents_to_qdrant(
        urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size
    ))
    
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
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
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
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        results[idx] = batch_contents[idx]
                
                contextual_contents = results
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the batch
        batch_embeddings = create_embeddings_batch(contextual_contents)
        
        # Create PointStruct objects
        qdrant_points = []
        for i, point in enumerate(points_batch):
            qdrant_points.append(PointStruct(
                id=point["id"],
                vector=batch_embeddings[i],
                payload=point["payload"]
            ))
        
        # Upsert batch to Qdrant
        try:
            client.upsert_points("crawled_pages", qdrant_points)
            print(f"Successfully inserted batch {batch_idx + 1}/{len(point_batches)}")
        except Exception as e:
            print(f"Error inserting batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("crawled_pages", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    print(f"Failed to insert individual point {point.id}: {individual_error}")
            
            if successful_inserts > 0:
                print(f"Successfully inserted {successful_inserts}/{len(qdrant_points)} points individually")

def search_documents(
    client: QdrantClientWrapper, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents using Qdrant vector similarity.
    
    Args:
        client: Qdrant client wrapper
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
        results = client.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
        return results
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks

def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Uses the chat model configured via CHAT_MODEL environment variable (with fallback to MODEL_CHOICE).
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    # Get chat model with fallback to legacy MODEL_CHOICE for backward compatibility
    model_choice = os.getenv("CHAT_MODEL", os.getenv("MODEL_CHOICE"))
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        client = get_chat_client()
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."

def add_code_examples_to_supabase(
    client: QdrantClientWrapper,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 100
):
    """
    Add code examples to Qdrant code_examples collection.
    LEGACY FUNCTION NAME: Maintained for compatibility.
    
    Args:
        client: Qdrant client wrapper
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return
    
    # Get point batches from Qdrant client (this handles URL deletion)
    point_batches = list(client.add_code_examples_to_qdrant(
        urls, chunk_numbers, code_examples, summaries, metadatas, batch_size
    ))
    
    # Process each batch
    for batch_idx, points_batch in enumerate(point_batches):
        # Create embeddings for combined text (code + summary)
        combined_texts = [point["combined_text"] for point in points_batch]
        embeddings = create_embeddings_batch(combined_texts)
        
        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for i, embedding in enumerate(embeddings):
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(f"Warning: Zero or invalid embedding detected, creating new one...")
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(combined_texts[i])
                valid_embeddings.append(single_embedding)
        
        # Create PointStruct objects
        qdrant_points = []
        for i, point in enumerate(points_batch):
            qdrant_points.append(PointStruct(
                id=point["id"],
                vector=valid_embeddings[i],
                payload=point["payload"]
            ))
        
        # Upsert batch to Qdrant
        try:
            client.upsert_points("code_examples", qdrant_points)
            print(f"Inserted batch {batch_idx + 1} of {len(point_batches)} code examples")
        except Exception as e:
            print(f"Error inserting code examples batch {batch_idx + 1}: {e}")
            # Try inserting points individually as fallback
            successful_inserts = 0
            for point in qdrant_points:
                try:
                    client.upsert_points("code_examples", [point])
                    successful_inserts += 1
                except Exception as individual_error:
                    print(f"Failed to insert individual code example {point.id}: {individual_error}")
            
            if successful_inserts > 0:
                print(f"Successfully inserted {successful_inserts}/{len(qdrant_points)} code examples individually")

def update_source_info(client: QdrantClientWrapper, source_id: str, summary: str, word_count: int):
    """
    Update source information using Qdrant client wrapper.
    
    Args:
        client: Qdrant client wrapper
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        client.update_source_info(source_id, summary, word_count)
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")

def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    Uses the chat model configured via CHAT_MODEL environment variable (with fallback to MODEL_CHOICE).
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get chat model with fallback to legacy MODEL_CHOICE for backward compatibility
    model_choice = os.getenv("CHAT_MODEL", os.getenv("MODEL_CHOICE"))
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        client = get_chat_client()
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary

def search_code_examples(
    client: QdrantClientWrapper, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples using Qdrant vector similarity.
    
    Args:
        client: Qdrant client wrapper
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    # Execute the search using Qdrant client
    try:
        results = client.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata,
            source_filter=source_id
        )
        return results
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []


# Device Management and Diagnostics
# Import device management utilities
try:
    from .device_manager import get_device_info as _get_device_info, cleanup_gpu_memory, get_optimal_device
except ImportError:
    from device_manager import get_device_info as _get_device_info, cleanup_gpu_memory, get_optimal_device


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for diagnostics.
    
    Provides information about available compute devices (CPU, CUDA, MPS)
    including memory status, device capabilities, and availability.
    
    Returns:
        Dict with device capabilities and status information including:
        - torch_available: Whether PyTorch is available
        - cuda_available: Whether CUDA GPUs are available
        - mps_available: Whether Apple Silicon MPS is available
        - device_count: Number of available CUDA devices
        - devices: List of detailed device information
    """
    return _get_device_info()


def log_device_status() -> None:
    """
    Log comprehensive device information for debugging and monitoring.
    
    Logs device information including available GPUs, memory status,
    and device capabilities. Useful for troubleshooting GPU acceleration issues.
    """
    device_info = get_device_info()
    
    print("=== Device Status Report ===")
    print(f"PyTorch Available: {device_info['torch_available']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"MPS Available: {device_info['mps_available']}")
    print(f"CUDA Device Count: {device_info['device_count']}")
    
    if device_info['devices']:
        print("Available Devices:")
        for device in device_info['devices']:
            if 'name' in device and 'type' not in device:  # CUDA device
                print(f"  - CUDA {device['index']}: {device['name']}")
                print(f"    Total Memory: {device['memory_total_gb']:.2f} GB")
                print(f"    Allocated Memory: {device['memory_allocated_gb']:.2f} GB")
                print(f"    Current Device: {device['is_current']}")
            elif 'type' in device:  # MPS device
                print(f"  - {device['name']} ({device['type']})")
    else:
        print("No GPU devices available")
    
    print("============================")


def monitor_gpu_memory() -> Optional[Dict[str, float]]:
    """
    Monitor GPU memory usage for the current device.
    
    Returns:
        Dict with memory information in GB, or None if CUDA not available:
        - allocated: Currently allocated memory
        - reserved: Reserved memory (includes allocated)
        - max_allocated: Peak allocated memory since last reset
        - max_reserved: Peak reserved memory since last reset
        - total: Total device memory
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        current_device = torch.cuda.current_device()
        memory_info = {
            'allocated': torch.cuda.memory_allocated(current_device) / (1024**3),
            'reserved': torch.cuda.memory_reserved(current_device) / (1024**3),
            'max_allocated': torch.cuda.max_memory_allocated(current_device) / (1024**3),
            'max_reserved': torch.cuda.max_memory_reserved(current_device) / (1024**3),
            'total': torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
        }
        return memory_info
    except Exception as e:
        print(f"Error monitoring GPU memory: {e}")
        return None


def log_gpu_memory_status() -> None:
    """
    Log current GPU memory status for monitoring and debugging.
    
    Provides detailed memory usage information including allocated,
    reserved, and peak usage statistics.
    """
    memory_info = monitor_gpu_memory()
    
    if memory_info is None:
        print("GPU memory monitoring not available (CUDA not detected)")
        return
    
    print("=== GPU Memory Status ===")
    print(f"Allocated: {memory_info['allocated']:.2f} GB")
    print(f"Reserved: {memory_info['reserved']:.2f} GB")
    print(f"Max Allocated: {memory_info['max_allocated']:.2f} GB")
    print(f"Max Reserved: {memory_info['max_reserved']:.2f} GB")
    print(f"Total Memory: {memory_info['total']:.2f} GB")
    print(f"Utilization: {(memory_info['allocated']/memory_info['total']*100):.1f}%")
    print("=========================")


def get_optimal_compute_device(preference: str = "auto") -> str:
    """
    Get the optimal compute device for machine learning operations.
    
    Provides a simple interface to device selection with fallback to CPU.
    This function wraps the more comprehensive device_manager functionality.
    
    Args:
        preference: Device preference - "auto", "cuda", "cpu", "mps"
        
    Returns:
        String representation of the optimal device (e.g., "cuda:0", "cpu")
    """
    try:
        device = get_optimal_device(preference)
        return str(device)
    except Exception as e:
        print(f"Error getting optimal device: {e}. Falling back to CPU.")
        return "cpu"


def cleanup_compute_memory() -> None:
    """
    Clean up compute memory (GPU cache) to prevent memory leaks.
    
    Safe wrapper around GPU memory cleanup that handles cases where
    GPU is not available. Should be called after intensive compute operations.
    """
    try:
        cleanup_gpu_memory()
    except Exception as e:
        print(f"Error during memory cleanup: {e}")


def health_check_gpu_acceleration() -> Dict[str, Any]:
    """
    Comprehensive health check for GPU acceleration capabilities.
    
    Performs actual device testing to verify GPU acceleration is working
    correctly. Useful for monitoring and troubleshooting deployment issues.
    
    Returns:
        Dict with health check results including:
        - gpu_available: Whether GPU is detected and working
        - device_name: Name of the GPU device
        - memory_available: Available GPU memory
        - test_passed: Whether GPU operations test passed
        - error_message: Error details if test failed
    """
    health_status = {
        'gpu_available': False,
        'device_name': 'CPU',
        'memory_available_gb': None,
        'test_passed': False,
        'error_message': None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Test actual GPU operations
            device = torch.device("cuda:0")
            
            # Perform test operation
            test_tensor = torch.randn(100, 100, device=device)
            result = test_tensor @ test_tensor.T
            
            # If we get here, GPU test passed
            health_status.update({
                'gpu_available': True,
                'device_name': torch.cuda.get_device_name(device),
                'memory_available_gb': torch.cuda.get_device_properties(device).total_memory / (1024**3),
                'test_passed': True
            })
            
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Test MPS (Apple Silicon)
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100, device=device)
            result = test_tensor.sum()
            
            health_status.update({
                'gpu_available': True,
                'device_name': 'Apple Silicon GPU (MPS)',
                'memory_available_gb': None,  # MPS doesn't expose memory info
                'test_passed': True
            })
            
    except Exception as e:
        health_status['error_message'] = str(e)
    
    return health_status