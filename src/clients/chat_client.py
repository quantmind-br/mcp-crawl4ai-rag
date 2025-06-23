"""
Chat model client with fallback support.

This module provides robust chat completion functionality with automatic fallback,
rate limiting, and circuit breaker patterns.
"""

import os
import time
import random
from typing import List, Dict, Any, Tuple, Optional

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


class ChatClient(BaseOpenAIClient):
    """Chat model client with automatic fallback support."""
    
    def get_primary_config(self) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
        """Get primary chat model configuration."""
        model = os.getenv("CHAT_MODEL")
        api_key = os.getenv("CHAT_MODEL_API_KEY")
        api_base = os.getenv("CHAT_MODEL_API_BASE")
        
        if not model or not api_key:
            raise ValueError("CHAT_MODEL and CHAT_MODEL_API_KEY must be set")
        
        client_args = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
        if api_base:
            client_args["base_url"] = api_base
            
        return model, api_key, api_base, client_args
    
    def get_fallback_config(self) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
        """Get fallback chat model configuration."""
        model = os.getenv("CHAT_MODEL_FALLBACK")
        api_key = os.getenv("CHAT_MODEL_FALLBACK_API_KEY")
        api_base = os.getenv("CHAT_MODEL_FALLBACK_API_BASE")
        
        client_args = {}
        if model and api_key:
            client_args = {"api_key": api_key, "timeout": REQUEST_TIMEOUT}
            if api_base:
                client_args["base_url"] = api_base
                
        return model, api_key, api_base, client_args
    
    def get_use_fallback_flag(self) -> bool:
        """Check if chat model fallback is enabled."""
        return os.getenv("USE_CHAT_MODEL_FALLBACK", "false").lower() == "true"
    
    def get_client_with_fallback(self) -> Tuple[openai.OpenAI, str, bool]:
        """
        Get a chat client with automatic fallback support using cached clients.
        
        Returns:
            Tuple containing:
            - OpenAI client instance (cached)
            - Model name being used
            - Boolean indicating if fallback was used (True = fallback, False = primary)
        """
        # Primary model configuration
        primary_model, primary_api_key, primary_api_base, primary_client_args = self.get_primary_config()
        
        # Try primary model first (with caching)
        primary_key = f"chat_primary_{primary_model}_{primary_api_base or 'default'}"
        
        def create_primary_client():
            print(f"Attempting to use primary chat model: {primary_model}")
            return openai.OpenAI(**primary_client_args)
        
        try:
            client = get_cached_client(primary_key, create_primary_client)
            print(f"✅ Primary chat model {primary_model} client ready")
            return client, primary_model, False
            
        except Exception as e:
            print(f"❌ Primary chat model client creation failed: {e}")
            record_error(primary_key, e)
        
        # Primary model failed, try fallback
        fallback_model, fallback_api_key, fallback_api_base, fallback_client_args = self.get_fallback_config()
        
        if fallback_model and fallback_api_key:
            fallback_key = f"chat_fallback_{fallback_model}_{fallback_api_base or 'default'}"
            
            def create_fallback_client():
                print(f"🔄 Attempting fallback to: {fallback_model}")
                return openai.OpenAI(**fallback_client_args)
            
            try:
                client = get_cached_client(fallback_key, create_fallback_client)
                print(f"✅ Fallback chat model {fallback_model} client ready")
                return client, fallback_model, True
                
            except Exception as e:
                print(f"❌ Fallback chat model client creation failed: {e}")
                record_error(fallback_key, e)
        else:
            print("⚠️  No fallback chat model configured")
        
        # Both primary and fallback failed
        raise Exception(f"Both primary ({primary_model}) and fallback ({fallback_model}) chat models failed")
    
    def make_completion_with_fallback(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        Make a chat completion with automatic fallback, rate limiting, and circuit breaker.
        
        Args:
            messages: List of message dictionaries for chat completion
            **kwargs: Additional arguments to pass to chat completion
            
        Returns:
            OpenAI chat completion response
        """
        # Get semaphore for concurrency control
        semaphore = get_global_semaphore()
        
        # Configuration  
        use_fallback = self.get_use_fallback_flag()
        max_retries = config.CHAT_MAX_RETRIES
        base_delay = config.RETRY_BASE_DELAY
        max_delay = config.RETRY_MAX_DELAY
        primary_error = None
        
        # Primary model configuration
        primary_model, _, primary_api_base, _ = self.get_primary_config()
        primary_key = f"chat_primary_{primary_model}_{primary_api_base or 'default'}"
        
        # Check circuit breaker for primary
        if is_circuit_breaker_open(primary_key):
            print(f"🚨 Circuit breaker open for primary model {primary_model}, forcing fallback")
            primary_error = Exception("Circuit breaker open")
        else:
            # Try primary model with rate limiting
            with semaphore:
                try:
                    rate_limit_delay(primary_key)
                    client, model_name, is_fallback = self.get_client_with_fallback()
                    
                    if not is_fallback:
                        # Attempt primary model with improved retry logic
                        for attempt in range(max_retries):
                            try:
                                response = client.chat.completions.create(
                                    model=model_name,
                                    messages=messages,
                                    **kwargs
                                )
                                record_success(primary_key)
                                if attempt > 0:
                                    print(f"✅ Primary model {model_name} succeeded on attempt {attempt + 1}")
                                return response
                                
                            except Exception as e:
                                if is_retryable_error(e) and attempt < max_retries - 1:
                                    # Exponential backoff with jitter
                                    jitter = random.uniform(0.5, 1.5)
                                    delay = min(base_delay * (2 ** attempt) * jitter, max_delay)
                                    print(f"⚠️  Primary model {model_name} failed (attempt {attempt + 1}/{max_retries}): {e}")
                                    print(f"🔄 Retrying in {delay:.1f}s...")
                                    time.sleep(delay)
                                    continue
                                else:
                                    print(f"❌ Primary model {model_name} failed definitively: {e}")
                                    record_error(primary_key, e)
                                    primary_error = e
                                    break
                                    
                except Exception as e:
                    print(f"❌ Primary model client failed: {e}")
                    record_error(primary_key, e)
                    primary_error = e
        
        # Try fallback if enabled and primary failed
        if use_fallback and primary_error:
            fallback_model, _, fallback_api_base, _ = self.get_fallback_config()
            fallback_key = f"chat_fallback_{fallback_model}_{fallback_api_base or 'default'}"
            
            if is_circuit_breaker_open(fallback_key):
                print(f"🚨 Circuit breaker open for fallback model {fallback_model}")
                raise Exception(f"Both primary and fallback models have circuit breakers open")
            
            # Try fallback with rate limiting
            with semaphore:
                try:
                    rate_limit_delay(fallback_key)
                    
                    # Get fallback client using cache
                    def create_fallback_client():
                        _, fallback_api_key, _, fallback_client_args = self.get_fallback_config()
                        if not fallback_model or not fallback_api_key:
                            raise Exception("Fallback model not configured")
                        return openai.OpenAI(**fallback_client_args)
                    
                    fallback_client = get_cached_client(fallback_key, create_fallback_client)
                    print(f"🔄 Using fallback model: {fallback_model}")
                    
                    # Attempt fallback with retry logic
                    for attempt in range(max_retries):
                        try:
                            response = fallback_client.chat.completions.create(
                                model=fallback_model,
                                messages=messages,
                                **kwargs
                            )
                            record_success(fallback_key)
                            print(f"✅ Fallback model {fallback_model} succeeded" + (f" on attempt {attempt + 1}" if attempt > 0 else ""))
                            return response
                            
                        except Exception as e:
                            if is_retryable_error(e) and attempt < max_retries - 1:
                                jitter = random.uniform(0.5, 1.5)
                                delay = min(base_delay * (2 ** attempt) * jitter, max_delay)
                                print(f"⚠️  Fallback model {fallback_model} failed (attempt {attempt + 1}/{max_retries}): {e}")
                                print(f"🔄 Retrying fallback in {delay:.1f}s...")
                                time.sleep(delay)
                                continue
                            else:
                                print(f"❌ Fallback model {fallback_model} failed definitively: {e}")
                                record_error(fallback_key, e)
                                raise e
                                
                except Exception as fallback_error:
                    record_error(fallback_key, fallback_error)
                    print(f"❌ Fallback model also failed: {fallback_error}")
                    raise Exception(f"Both primary and fallback chat models failed. Primary: {primary_error}. Fallback: {fallback_error}")
        
        # No fallback available
        if not use_fallback:
            print("⚠️  Chat model fallback is disabled (USE_CHAT_MODEL_FALLBACK=false)")
        
        if primary_error:
            raise primary_error
        
        raise Exception("Unexpected error in chat completion system")