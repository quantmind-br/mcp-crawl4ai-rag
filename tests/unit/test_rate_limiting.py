"""
Unit tests for rate limiting utilities.
"""

import time
import threading
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.utils.rate_limiting import (
    get_global_semaphore,
    rate_limit_delay,
    is_circuit_breaker_open,
    record_error,
    record_success,
    get_cached_client,
    is_retryable_error,
    _client_cache,
    _consecutive_errors,
    _circuit_breaker_until,
    _last_request_time
)


class TestRateLimiting:
    """Test suite for rate limiting functionality."""
    
    def setup_method(self):
        """Clear global state before each test."""
        global _client_cache, _consecutive_errors, _circuit_breaker_until, _last_request_time
        _client_cache.clear()
        _consecutive_errors.clear()
        _circuit_breaker_until.clear()
        _last_request_time.clear()
    
    def test_get_global_semaphore(self):
        """Test that global semaphore is created and reused."""
        sem1 = get_global_semaphore()
        sem2 = get_global_semaphore()
        
        assert sem1 is sem2
        assert isinstance(sem1, threading.Semaphore)
    
    @patch('time.sleep')
    @patch('time.time')
    def test_rate_limit_delay(self, mock_time, mock_sleep):
        """Test rate limiting delay functionality."""
        # Setup time progression
        mock_time.side_effect = [1000.0, 1000.2, 1000.5]  # Current time calls
        
        # First call should not delay (no previous request)
        rate_limit_delay("test_endpoint")
        mock_sleep.assert_not_called()
        
        # Second call should delay (too soon after first)
        rate_limit_delay("test_endpoint")
        mock_sleep.assert_called_once()
        
        # Verify delay calculation
        expected_delay = 0.5 - 0.2  # RATE_LIMIT_DELAY - time_diff
        mock_sleep.assert_called_with(expected_delay)
    
    def test_is_retryable_error(self):
        """Test retryable error detection."""
        # Test retryable errors
        retryable_errors = [
            Exception("HTTP 503 Service Unavailable"),
            Exception("HTTP 429 Rate Limit Exceeded"),
            Exception("HTTP 500 Internal Server Error"),
            Exception("Connection timeout"),
            Exception("Service unavailable"),
            Exception("Rate limit exceeded")
        ]
        
        for error in retryable_errors:
            assert is_retryable_error(error) is True
        
        # Test non-retryable errors
        non_retryable_errors = [
            Exception("HTTP 404 Not Found"),
            Exception("Invalid API key"),
            Exception("Syntax error"),
            Exception("Unknown error")
        ]
        
        for error in non_retryable_errors:
            assert is_retryable_error(error) is False
    
    def test_record_error_and_circuit_breaker(self):
        """Test error recording and circuit breaker triggering."""
        endpoint = "test_endpoint"
        retryable_error = Exception("HTTP 503 Service Unavailable")
        non_retryable_error = Exception("HTTP 404 Not Found")
        
        # Test that non-retryable errors don't increment counter
        record_error(endpoint, non_retryable_error)
        assert _consecutive_errors.get(endpoint, 0) == 0
        
        # Test retryable error increments counter
        record_error(endpoint, retryable_error)
        assert _consecutive_errors[endpoint] == 1
        
        # Test circuit breaker doesn't trigger until threshold
        assert is_circuit_breaker_open(endpoint) is False
        
        # Add more errors to trigger circuit breaker
        record_error(endpoint, retryable_error)
        record_error(endpoint, retryable_error)
        
        # Circuit breaker should now be open
        assert is_circuit_breaker_open(endpoint) is True
        assert endpoint in _circuit_breaker_until
    
    def test_record_success_resets_counter(self):
        """Test that recording success resets error counter."""
        endpoint = "test_endpoint"
        error = Exception("HTTP 503 Service Unavailable")
        
        # Build up error count
        record_error(endpoint, error)
        record_error(endpoint, error)
        assert _consecutive_errors[endpoint] == 2
        
        # Success should reset counter
        record_success(endpoint)
        assert _consecutive_errors[endpoint] == 0
    
    @patch('time.time')
    def test_circuit_breaker_timeout(self, mock_time):
        """Test that circuit breaker resets after timeout."""
        endpoint = "test_endpoint"
        error = Exception("HTTP 503 Service Unavailable")
        
        # Trigger circuit breaker
        for _ in range(3):
            record_error(endpoint, error)
        
        assert is_circuit_breaker_open(endpoint) is True
        
        # Simulate time passing (6 minutes > 5 minute timeout)
        with patch('src.utils.rate_limiting.datetime') as mock_datetime:
            future_time = datetime.now() + timedelta(minutes=6)
            mock_datetime.now.return_value = future_time
            
            # Circuit breaker should be closed now
            assert is_circuit_breaker_open(endpoint) is False
            assert _consecutive_errors[endpoint] == 0
    
    @patch('time.time')
    def test_get_cached_client(self, mock_time):
        """Test client caching functionality."""
        mock_time.return_value = 1000.0
        
        # Mock client creation function
        mock_client = MagicMock()
        create_func = MagicMock(return_value=mock_client)
        
        # First call should create client
        client1 = get_cached_client("test_key", create_func)
        assert client1 is mock_client
        create_func.assert_called_once()
        
        # Second call should return cached client
        create_func.reset_mock()
        client2 = get_cached_client("test_key", create_func)
        assert client2 is mock_client
        create_func.assert_not_called()
        
        # Test cache expiration
        mock_time.return_value = 1000.0 + 3700  # 1 hour + 100 seconds
        create_func.reset_mock()
        
        client3 = get_cached_client("test_key", create_func)
        create_func.assert_called_once()  # Should create new client
    
    def test_different_endpoints_isolated(self):
        """Test that different endpoints have isolated state."""
        endpoint1 = "endpoint1"
        endpoint2 = "endpoint2"
        error = Exception("HTTP 503 Service Unavailable")
        
        # Trigger circuit breaker for endpoint1
        for _ in range(3):
            record_error(endpoint1, error)
        
        # endpoint1 should have circuit breaker open
        assert is_circuit_breaker_open(endpoint1) is True
        
        # endpoint2 should not be affected
        assert is_circuit_breaker_open(endpoint2) is False
        assert _consecutive_errors.get(endpoint2, 0) == 0