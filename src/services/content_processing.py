"""
Content processing service.

This module provides services for processing content, including contextual embeddings,
code example generation, and source summarization.
"""

import concurrent.futures
from typing import List, Dict, Any, Tuple

from ..clients.chat_client import ChatClient
from ..utils.content_utils import extract_code_blocks
from ..config import config


class ContentProcessingService:
    """Service for content processing operations."""
    
    def __init__(self):
        self.chat_client = ChatClient()
    
    def generate_contextual_embedding(self, full_document: str, chunk: str) -> Tuple[str, bool]:
        """
        Generate contextual information for a chunk within a document to improve retrieval.
        
        Args:
            full_document: The complete document text
            chunk: The specific chunk of text to generate context for
            
        Returns:
            Tuple containing:
            - The contextual text that situates the chunk within the document
            - Boolean indicating if contextual embedding was performed
        """
        try:
            # Create the prompt for generating contextual information
            prompt = f"""<document>\n{full_document}\n</document>\n<chunk>\n{chunk}\n</chunk> \nPlease give a short succinct context to situate this chunk within the whole document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

            # Use robust chat completion with automatic fallback
            response = self.chat_client.make_completion_with_fallback(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the generated context
            context = response.choices[0].message.content.strip()
            
            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"
            
            return contextual_text, True
        
        except Exception as e:
            print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
            return chunk, False
    
    def process_chunk_with_context(self, args):
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
        return self.generate_contextual_embedding(full_document, content)
    
    def generate_code_example_summary(self, code: str, context_before: str, context_after: str) -> str:
        """
        Generate a summary for a code example using its surrounding context.
        
        Args:
            code: The code example
            context_before: Context before the code
            context_after: Context after the code
            
        Returns:
            A summary of what the code example demonstrates
        """
        # Create the prompt
        prompt = f"""<context_before>\n{context_before}\n</context_before>\n<code_example>\n{code}\n</code_example>\n<context_after>\n{context_after}\n</context_after>\n\nPlease provide a concise summary of what this code example demonstrates.\n"""
        
        try:
            # Use the new fallback system
            response = self.chat_client.make_completion_with_fallback(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating code example summary: {e}")
            return "Code example for demonstration purposes."
    
    def process_code_example(self, args):
        """
        Process a single code example to generate its summary.
        This function is designed to be used with concurrent.futures.
        
        Args:
            args: Tuple containing (code, context_before, context_after)
            
        Returns:
            The generated summary
        """
        code, context_before, context_after = args
        return self.generate_code_example_summary(code, context_before, context_after)
    
    def process_code_examples_parallel(self, code_blocks: List[Dict[str, Any]]) -> List[str]:
        """
        Process multiple code examples in parallel to generate summaries.
        
        Args:
            code_blocks: List of code block dictionaries
            
        Returns:
            List of generated summaries
        """
        if not code_blocks:
            return []
        
        # Process code examples in parallel
        max_workers = config.MAX_WORKERS_SUMMARY
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for parallel processing
            summary_args = [(block['code'], block['context_before'], block['context_after']) 
                            for block in code_blocks]
            
            # Generate summaries in parallel
            summaries = list(executor.map(self.process_code_example, summary_args))
        
        return summaries
    
    def extract_source_summary(self, source_id: str, content: str, max_length: int = None) -> str:
        """
        Extract a summary for a source from its content using an LLM.
        
        This function uses the OpenAI API to generate a concise summary of the source content.
        
        Args:
            source_id: The source ID (domain)
            content: The content to extract a summary from
            max_length: Maximum length of the summary
            
        Returns:
            A summary string
        """
        # Use configuration for max_length if not provided
        if max_length is None:
            max_length = config.SOURCE_SUMMARY_MAX_LENGTH
        
        # Default summary if we can't extract anything meaningful
        default_summary = f"Content from {source_id}"
        
        if not content or len(content.strip()) == 0:
            return default_summary
        
        # Limit content length to avoid token limits
        truncation_limit = config.CONTENT_TRUNCATION_LIMIT
        truncated_content = content[:truncation_limit] if len(content) > truncation_limit else content
        
        prompt = f"""<library_or_tool_content>\n{truncated_content}\n</library_or_tool_content>\n\nPlease provide a concise summary of the above library, tool, or framework.\n"""
        
        try:
            # Use robust chat completion with automatic fallback
            response = self.chat_client.make_completion_with_fallback(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                    {"role": "user", "content": prompt}
                ]
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
    
    def process_source_summaries_parallel(self, source_content_map: Dict[str, str]) -> Dict[str, str]:
        """
        Process multiple source summaries in parallel.
        
        Args:
            source_content_map: Dictionary mapping source_id to content
            
        Returns:
            Dictionary mapping source_id to summary
        """
        if not source_content_map:
            return {}
        
        max_workers_source_summary = config.MAX_WORKERS_SOURCE_SUMMARY
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_source_summary) as executor:
            source_summary_args = [(source_id, content) for source_id, content in source_content_map.items()]
            source_summaries = list(executor.map(lambda args: self.extract_source_summary(args[0], args[1]), source_summary_args))
        
        # Create result dictionary
        result = {}
        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            result[source_id] = summary
        
        return result