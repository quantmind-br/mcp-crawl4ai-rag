"""
Rate limiting and circuit breaker functionality.

This module provides comprehensive rate limiting, circuit breaker patterns,
and client caching for preventing API overload and 503 errors.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Callable

from ..config import config


# Global state for rate limiting and circuit breaker
_client_cache = {}
_client_cache_lock = threading.Lock()
_request_semaphore = None
_last_request_time = {}
_consecutive_errors = {}
_circuit_breaker_until = {}

# Configuration
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.5"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
CLIENT_CACHE_TTL = int(os.getenv("CLIENT_CACHE_TTL", "3600"))  # 1 hour


def get_global_semaphore():
    """Get or create global request semaphore."""
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _request_semaphore


def rate_limit_delay(endpoint_key: str):
    """Implement rate limiting with minimum delay between requests."""
    global _last_request_time
    
    current_time = time.time()
    last_time = _last_request_time.get(endpoint_key, 0)
    
    time_diff = current_time - last_time
    if time_diff < RATE_LIMIT_DELAY:
        delay = RATE_LIMIT_DELAY - time_diff
        print(f"⏱️  Rate limiting: sleeping {delay:.2f}s for {endpoint_key}")
        time.sleep(delay)
    
    _last_request_time[endpoint_key] = time.time()


def is_circuit_breaker_open(endpoint_key: str) -> bool:
    """Check if circuit breaker is open for given endpoint."""
    global _circuit_breaker_until
    
    if endpoint_key in _circuit_breaker_until:
        if datetime.now() < _circuit_breaker_until[endpoint_key]:
            return True
        else:
            # Circuit breaker timeout expired, reset
            del _circuit_breaker_until[endpoint_key]
            _consecutive_errors[endpoint_key] = 0
    
    return False


def record_error(endpoint_key: str, error: Exception):
    """Record error and potentially trigger circuit breaker."""
    global _consecutive_errors, _circuit_breaker_until
    
    if is_retryable_error(error):
        _consecutive_errors[endpoint_key] = _consecutive_errors.get(endpoint_key, 0) + 1
        
        if _consecutive_errors[endpoint_key] >= CIRCUIT_BREAKER_THRESHOLD:
            # Open circuit breaker for configured timeout
            timeout_minutes = config.CIRCUIT_BREAKER_TIMEOUT_MINUTES
            _circuit_breaker_until[endpoint_key] = datetime.now() + timedelta(minutes=timeout_minutes)
            print(f"🚨 Circuit breaker opened for {endpoint_key} after {_consecutive_errors[endpoint_key]} consecutive errors")


def record_success(endpoint_key: str):
    """Record successful request and reset error counter."""
    global _consecutive_errors
    _consecutive_errors[endpoint_key] = 0


def get_cached_client(client_key: str, create_func: Callable) -> Any:
    """Get or create cached client with TTL."""
    global _client_cache
    
    with _client_cache_lock:
        current_time = time.time()
        
        # Check if we have a valid cached client
        if client_key in _client_cache:
            client_data = _client_cache[client_key]
            if current_time - client_data['created_at'] < CLIENT_CACHE_TTL:
                print(f"♻️  Using cached client for {client_key}")
                return client_data['client']
            else:
                print(f"⏰ Cache expired for {client_key}, creating new client")
                del _client_cache[client_key]
        
        # Create new client
        print(f"🔧 Creating new client for {client_key}")
        client = create_func()
        _client_cache[client_key] = {
            'client': client,
            'created_at': current_time
        }
        
        return client


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable (temporary) or should trigger fallback.
    
    Args:
        error: Exception that occurred during API call
        
    Returns:
        True if error is retryable, False if should trigger fallback
    """
    error_str = str(error).lower()
    
    # HTTP status codes that indicate temporary server issues
    retryable_codes = ['503', '429', '500', '502', '504']
    retryable_messages = [
        'service unavailable',
        'rate limit',
        'timeout',
        'connection error',
        'no available server',
        'server error'
    ]
    
    # Check for specific HTTP status codes
    for code in retryable_codes:
        if code in error_str:
            return True
    
    # Check for specific error messages
    for message in retryable_messages:
        if message in error_str:
            return True
    
    return False