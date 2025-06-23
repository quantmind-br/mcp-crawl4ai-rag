"""
Embedding model client with fallback support.

This module provides robust embedding functionality with automatic fallback,
rate limiting, circuit breaker patterns, and dimension handling.
"""

import os
import time
import random
from typing import List, Tuple, Optional, Dict, Any

import openai

from .base import BaseOpenAIClient
from ..config import config
from ..utils.rate_limiting import (
    get_global_semaphore,
    rate_limit_delay,
    is_circuit_breaker_open,
    record_error,
    record_success,
    get_cached_client,
    is_retryable_error,
    REQUEST_TIMEOUT
)


class EmbeddingClient(BaseOpenAIClient):
    """Embedding model client with automatic fallback support."""
    
    def get_primary_config(self) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
        """Get primary embedding model configuration."""
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.getenv("EMBEDDING_MODEL_API_KEY")
        api_base = os.getenv("EMBEDDING_MODEL_API_BASE")
        
        if not api_key:
            raise ValueError("EMBEDDING_MODEL_API_KEY must be set")
        
        client_args = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
        if api_base:
            client_args["base_url"] = api_base
            
        return model, api_key, api_base, client_args
    
    def get_fallback_config(self) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
        """Get fallback embedding model configuration."""
        model = os.getenv("EMBEDDING_MODEL_FALLBACK")
        api_key = os.getenv("EMBEDDING_MODEL_FALLBACK_API_KEY")
        api_base = os.getenv("EMBEDDING_MODEL_FALLBACK_API_BASE")
        
        client_args = {}
        if model and api_key:
            client_args = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
            if api_base:
                client_args["base_url"] = api_base
                
        return model, api_key, api_base, client_args
    
    def get_use_fallback_flag(self) -> bool:
        """Check if embedding model fallback is enabled."""
        return os.getenv("USE_EMBEDDING_MODEL_FALLBACK", "false").lower() == "true"
    
    def get_dimensions(self, is_fallback: bool = False) -> int:
        """Get embedding dimensions for primary or fallback model."""
        if is_fallback:
            return int(os.getenv("EMBEDDING_DIMENSIONS_FALLBACK", "1536"))
        else:
            return int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    
    def get_client_with_fallback(self) -> Tuple[openai.OpenAI, str, int, bool]:
        """
        Get an embedding client with automatic fallback support using cached clients.
        
        Returns:
            Tuple containing:
            - OpenAI client instance (cached)
            - Model name being used
            - Embedding dimensions
            - Boolean indicating if fallback was used (True = fallback, False = primary)
        """
        # Primary model configuration
        primary_model, primary_api_key, primary_api_base, primary_client_args = self.get_primary_config()
        primary_dims = self.get_dimensions(is_fallback=False)
        
        # Try primary model first (with caching)
        primary_key = f"embedding_primary_{primary_model}_{primary_api_base or 'default'}"
        
        def create_primary_embedding_client():
            print(f"Attempting to use primary embedding model: {primary_model}")
            return openai.OpenAI(**primary_client_args)
        
        try:
            client = get_cached_client(primary_key, create_primary_embedding_client)
            print(f"✅ Primary embedding model {primary_model} client ready")
            return client, primary_model, primary_dims, False
            
        except Exception as e:
            print(f"❌ Primary embedding model client creation failed: {e}")
            record_error(primary_key, e)
        
        # Primary model failed, try fallback
        fallback_model, fallback_api_key, fallback_api_base, fallback_client_args = self.get_fallback_config()
        fallback_dims = self.get_dimensions(is_fallback=True)
        
        if fallback_model and fallback_api_key:
            fallback_key = f"embedding_fallback_{fallback_model}_{fallback_api_base or 'default'}"
            
            def create_fallback_embedding_client():
                print(f"🔄 Attempting embedding fallback to: {fallback_model}")
                return openai.OpenAI(**fallback_client_args)
            
            try:
                client = get_cached_client(fallback_key, create_fallback_embedding_client)
                print(f"✅ Fallback embedding model {fallback_model} client ready")
                return client, fallback_model, fallback_dims, True
                
            except Exception as e:
                print(f"❌ Fallback embedding model client creation failed: {e}")
                record_error(fallback_key, e)
        else:
            print("⚠️  No fallback embedding model configured")
        
        # Both primary and fallback failed
        raise Exception(f"Both primary ({primary_model}) and fallback ({fallback_model}) embedding models failed")
    
    def handle_dimension_mismatch(self, embeddings: List[List[float]], target_dims: int) -> List[List[float]]:
        """
        Handle dimension mismatches between different embedding models.
        
        Args:
            embeddings: List of embedding vectors
            target_dims: Target dimension count
            
        Returns:
            List of embeddings adjusted to target dimensions
        """
        if not embeddings or not embeddings[0]:
            return embeddings
        
        current_dims = len(embeddings[0])
        
        if current_dims == target_dims:
            return embeddings
        elif current_dims > target_dims:
            # Truncate to target dimensions (keep most informative dimensions)
            print(f"📏 Truncating embeddings: {current_dims}D → {target_dims}D")
            return [embedding[:target_dims] for embedding in embeddings]
        else:
            # Pad with zeros to reach target dimensions
            print(f"📏 Padding embeddings: {current_dims}D → {target_dims}D")
            padding_size = target_dims - current_dims
            return [embedding + [0.0] * padding_size for embedding in embeddings]
    
    def create_embeddings_with_fallback(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings with rate limiting, circuit breaker, and automatic fallback.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
        
        # Get semaphore for concurrency control
        semaphore = get_global_semaphore()
        
        # Configuration
        use_fallback = self.get_use_fallback_flag()
        max_retries = config.EMBEDDING_MAX_RETRIES
        base_delay = config.RETRY_BASE_DELAY
        max_delay = config.RETRY_MAX_DELAY
        target_dims = self.get_dimensions(is_fallback=False)
        primary_error = None
        
        # Primary model configuration
        primary_model, _, primary_api_base, _ = self.get_primary_config()
        primary_key = f"embedding_primary_{primary_model}_{primary_api_base or 'default'}"
        
        # Check circuit breaker for primary
        if is_circuit_breaker_open(primary_key):
            print(f"🚨 Circuit breaker open for primary embedding model {primary_model}, forcing fallback")
            primary_error = Exception("Circuit breaker open")
        else:
            # Try primary model with rate limiting
            with semaphore:
                try:
                    rate_limit_delay(primary_key)
                    client, model_name, model_dims, is_fallback = self.get_client_with_fallback()
                    
                    if not is_fallback:
                        # Attempt primary model with improved retry logic
                        for attempt in range(max_retries):
                            try:
                                # Process texts for Qwen3-Embedding compatibility
                                processed_texts = texts
                                is_ollama = primary_api_base and "11434" in primary_api_base
                                if is_ollama and "qwen3-embedding" in model_name.lower():
                                    processed_texts = [text + "<|endoftext|>" for text in texts]
                                    print(f"Added <|endoftext|> token to {len(texts)} texts for Qwen3-Embedding compatibility")
                                
                                # Create embedding request
                                response = client.embeddings.create(
                                    model=model_name,
                                    input=processed_texts
                                )
                                
                                embeddings = [item.embedding for item in response.data]
                                embeddings = self.handle_dimension_mismatch(embeddings, target_dims)
                                
                                record_success(primary_key)
                                if attempt > 0:
                                    print(f"✅ Primary embedding model {model_name} succeeded on attempt {attempt + 1}")
                                return embeddings
                                
                            except Exception as e:
                                if is_retryable_error(e) and attempt < max_retries - 1:
                                    # Exponential backoff with jitter
                                    jitter = random.uniform(0.5, 1.5)
                                    delay = min(base_delay * (2 ** attempt) * jitter, max_delay)
                                    print(f"⚠️  Primary embedding model {model_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                                    print(f"🔄 Retrying in {delay:.1f}s...")
                                    time.sleep(delay)
                                    continue
                                else:
                                    print(f"❌ Primary embedding model {model_name} failed definitively: {e}")
                                    record_error(primary_key, e)
                                    primary_error = e
                                    break
                                    
                except Exception as e:
                    print(f"❌ Primary embedding model client failed: {e}")
                    record_error(primary_key, e)
                    primary_error = e
        
        # Try fallback if enabled and primary failed
        if use_fallback and primary_error:
            fallback_model, _, fallback_api_base, _ = self.get_fallback_config()
            fallback_key = f"embedding_fallback_{fallback_model}_{fallback_api_base or 'default'}"
            
            if is_circuit_breaker_open(fallback_key):
                print(f"🚨 Circuit breaker open for fallback embedding model {fallback_model}")
                raise Exception(f"Both primary and fallback embedding models have circuit breakers open")
            
            # Try fallback with rate limiting
            with semaphore:
                try:
                    rate_limit_delay(fallback_key)
                    
                    # Get fallback client using cache
                    def create_fallback_embedding_client():
                        _, fallback_api_key, _, fallback_client_args = self.get_fallback_config()
                        if not fallback_model or not fallback_api_key:
                            raise Exception("Fallback embedding model not configured")
                        return openai.OpenAI(**fallback_client_args)
                    
                    fallback_client = get_cached_client(fallback_key, create_fallback_embedding_client)
                    print(f"🔄 Using fallback embedding model: {fallback_model}")
                    
                    # Attempt fallback with retry logic
                    for attempt in range(max_retries):
                        try:
                            response = fallback_client.embeddings.create(
                                model=fallback_model,
                                input=texts
                            )
                            
                            embeddings = [item.embedding for item in response.data]
                            embeddings = self.handle_dimension_mismatch(embeddings, target_dims)
                            
                            record_success(fallback_key)
                            print(f"✅ Fallback embedding model {fallback_model} succeeded" + (f" on attempt {attempt + 1}" if attempt > 0 else ""))
                            return embeddings
                            
                        except Exception as e:
                            if is_retryable_error(e) and attempt < max_retries - 1:
                                jitter = random.uniform(0.5, 1.5)
                                delay = min(base_delay * (2 ** attempt) * jitter, max_delay)
                                print(f"⚠️  Fallback embedding model {fallback_model} failed (attempt {attempt + 1}/{max_retries}): {e}")
                                print(f"🔄 Retrying fallback in {delay:.1f}s...")
                                time.sleep(delay)
                                continue
                            else:
                                print(f"❌ Fallback embedding model {fallback_model} failed definitively: {e}")
                                record_error(fallback_key, e)
                                raise e
                                
                except Exception as fallback_error:
                    record_error(fallback_key, fallback_error)
                    print(f"❌ Fallback embedding model also failed: {fallback_error}")
                    raise Exception(f"Both primary and fallback embedding models failed. Primary: {primary_error}. Fallback: {fallback_error}")
        
        # No fallback available
        if not use_fallback:
            print("⚠️  Embedding model fallback is disabled (USE_EMBEDDING_MODEL_FALLBACK=false)")
        
        if primary_error:
            raise primary_error
            
        raise Exception("Unexpected error in embedding fallback system")
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a single API call.
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []
        
        try:
            return self.create_embeddings_with_fallback(texts)
        except Exception as e:
            print(f"Both primary and fallback embedding models failed: {e}")
            # Final fallback: return zero embeddings
            embedding_dims = self.get_dimensions(is_fallback=False)
            print(f"⚠️  Returning zero embeddings as final fallback")
            return [[0.0] * embedding_dims] * len(texts)
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        embedding_dims = self.get_dimensions(is_fallback=False)
        try:
            embeddings = self.create_embeddings_batch([text])
            return embeddings[0] if embeddings else [0.0] * embedding_dims
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return empty embedding if there's an error
            return [0.0] * embedding_dims