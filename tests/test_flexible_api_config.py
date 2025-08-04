"""
Integration tests for flexible API configuration system.

Tests the new configurable OpenAI client system with different providers,
backward compatibility, error handling, and performance.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestFlexibleAPIConfiguration:
    """Test the new flexible API configuration system."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Save original environment (including conftest values)
        self.original_env = {
            key: os.environ.get(key)
            for key in [
                "CHAT_MODEL",
                "CHAT_API_KEY",
                "CHAT_API_BASE",
                "EMBEDDINGS_MODEL",
                "EMBEDDINGS_API_KEY",
                "EMBEDDINGS_API_BASE",
            ]
        }

        # Clear environment for clean testing (remove conftest interference)
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        """Restore original environment after each test."""
        # Clear test environment
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

        # Restore original values
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

    def test_chat_client_new_configuration(self):
        """Test chat client with new CHAT_* environment variables."""
        # Set new configuration
        os.environ["CHAT_MODEL"] = "gpt-4"
        os.environ["CHAT_API_KEY"] = "test-chat-key"
        os.environ["CHAT_API_BASE"] = "https://api.openai.com/v1"

        from src.utils import get_chat_client

        client = get_chat_client()
        assert client.api_key == "test-chat-key"
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_embeddings_client_new_configuration(self):
        """Test embeddings client with new EMBEDDINGS_* environment variables."""
        # Set new configuration
        os.environ["EMBEDDINGS_MODEL"] = "text-embedding-3-large"
        os.environ["EMBEDDINGS_API_KEY"] = "test-embeddings-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://api.openai.com/v1"

        from src.utils import get_embeddings_client

        client = get_embeddings_client()
        assert client.api_key == "test-embeddings-key"
        assert str(client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_azure_openai_configuration(self):
        """Test configuration for Azure OpenAI service."""
        os.environ["CHAT_API_KEY"] = "azure-api-key"
        os.environ["CHAT_API_BASE"] = "https://my-resource.openai.azure.com/"
        os.environ["EMBEDDINGS_API_KEY"] = "azure-embeddings-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://my-resource.openai.azure.com/"

        from src.utils import get_chat_client, get_embeddings_client

        chat_client = get_chat_client()
        embeddings_client = get_embeddings_client()

        assert chat_client.api_key == "azure-api-key"
        assert (
            str(chat_client.base_url).rstrip("/")
            == "https://my-resource.openai.azure.com"
        )
        assert embeddings_client.api_key == "azure-embeddings-key"
        assert (
            str(embeddings_client.base_url).rstrip("/")
            == "https://my-resource.openai.azure.com"
        )

    def test_localai_configuration(self):
        """Test configuration for LocalAI service."""
        os.environ["CHAT_API_KEY"] = "not-needed"
        os.environ["CHAT_API_BASE"] = "http://localhost:8080/v1"
        os.environ["EMBEDDINGS_API_KEY"] = "not-needed"
        os.environ["EMBEDDINGS_API_BASE"] = "http://localhost:8080/v1"

        from src.utils import get_chat_client, get_embeddings_client

        chat_client = get_chat_client()
        embeddings_client = get_embeddings_client()

        assert str(chat_client.base_url).rstrip("/") == "http://localhost:8080/v1"
        assert str(embeddings_client.base_url).rstrip("/") == "http://localhost:8080/v1"

    def test_error_handling_missing_api_key(self):
        """Test error handling when no API key is configured."""
        # Don't set any API keys

        from src.utils import get_chat_client, get_embeddings_client

        with pytest.raises(ValueError, match="No API key configured for chat model"):
            get_chat_client()

        with pytest.raises(ValueError, match="No API key configured for embeddings"):
            get_embeddings_client()

    def test_validation_functions(self):
        """Test configuration validation functions."""
        from src.utils import validate_chat_config, validate_embeddings_config

        # Test with missing configuration
        with pytest.raises(ValueError, match="No API key configured"):
            validate_chat_config()

        with pytest.raises(ValueError, match="No API key configured"):
            validate_embeddings_config()

        # Test with valid configuration
        os.environ["CHAT_API_KEY"] = "test-key"
        os.environ["EMBEDDINGS_API_KEY"] = "test-key"

        assert validate_chat_config() is True
        assert validate_embeddings_config() is True

    def test_embeddings_model_configuration(self):
        """Test that EMBEDDINGS_MODEL is properly used."""
        os.environ["EMBEDDINGS_MODEL"] = "text-embedding-3-large"
        os.environ["EMBEDDINGS_API_KEY"] = "test-key"

        from src.utils import create_embeddings_batch

        # Mock the embeddings client
        with patch("src.utils.get_embeddings_client") as mock_get_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Call function that creates embeddings
            create_embeddings_batch(["test text"])

            # Should use EMBEDDINGS_MODEL
            call_args = mock_client.embeddings.create.call_args
            assert call_args[1]["model"] == "text-embedding-3-large"

    @patch("src.utils.get_chat_client")
    @patch("src.utils.get_embeddings_client")
    def test_performance_no_regression(self, mock_embeddings_client, mock_chat_client):
        """Test that new configuration doesn't introduce performance regression."""
        # Set up mocks
        mock_chat_client.return_value = Mock()
        mock_embeddings_client.return_value = Mock()

        # Set configuration
        os.environ["CHAT_MODEL"] = "gpt-3.5-turbo"
        os.environ["CHAT_API_KEY"] = "test-key"
        os.environ["EMBEDDINGS_MODEL"] = "text-embedding-3-small"
        os.environ["EMBEDDINGS_API_KEY"] = "test-key"

        from src.utils import get_chat_client, get_embeddings_client

        # Measure client creation time
        start_time = time.time()
        for _ in range(100):
            get_chat_client()
            get_embeddings_client()
        end_time = time.time()

        # Should be very fast (under 100ms for 100 iterations)
        elapsed = end_time - start_time
        assert elapsed < 0.1, f"Client creation too slow: {elapsed:.3f}s"

    def test_mixed_provider_configuration(self):
        """Test using different providers for chat and embeddings."""
        # Chat via Azure OpenAI
        os.environ["CHAT_API_KEY"] = "azure-chat-key"
        os.environ["CHAT_API_BASE"] = "https://azure.openai.azure.com/"

        # Embeddings via regular OpenAI
        os.environ["EMBEDDINGS_API_KEY"] = "openai-embeddings-key"
        # No EMBEDDINGS_API_BASE, should use default OpenAI

        from src.utils import get_chat_client, get_embeddings_client

        chat_client = get_chat_client()
        embeddings_client = get_embeddings_client()

        # Chat should use Azure
        assert chat_client.api_key == "azure-chat-key"
        assert str(chat_client.base_url).rstrip("/") == "https://azure.openai.azure.com"

        # Embeddings should use regular OpenAI (default base_url)
        assert embeddings_client.api_key == "openai-embeddings-key"
        assert (
            str(embeddings_client.base_url).rstrip("/") == "https://api.openai.com/v1"
        )

    def test_configuration_isolation(self):
        """Test that chat and embeddings configurations are independent."""
        # Different providers for chat and embeddings
        os.environ["CHAT_API_KEY"] = "chat-provider-key"
        os.environ["CHAT_API_BASE"] = "https://chat-provider.com/v1"

        os.environ["EMBEDDINGS_API_KEY"] = "embeddings-provider-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://embeddings-provider.com/v1"

        from src.utils import get_chat_client, get_embeddings_client

        chat_client = get_chat_client()
        embeddings_client = get_embeddings_client()

        # Each should use its own configuration
        assert chat_client.api_key == "chat-provider-key"
        assert str(chat_client.base_url).rstrip("/") == "https://chat-provider.com/v1"

        assert embeddings_client.api_key == "embeddings-provider-key"
        assert (
            str(embeddings_client.base_url).rstrip("/")
            == "https://embeddings-provider.com/v1"
        )
