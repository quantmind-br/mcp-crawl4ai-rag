"""
Tests for the LLM API client.
"""

import pytest
import os
from unittest.mock import Mock, patch
import openai
from src.clients.llm_api_client import (
    get_chat_client,
    get_embeddings_client,
    get_chat_fallback_client,
    get_embeddings_fallback_client,
    get_adaptive_chat_client,
    validate_chat_config,
    validate_chat_fallback_config,
    validate_embeddings_fallback_config,
    get_effective_fallback_config,
)


class TestLLMAPIClient:
    """Test cases for the LLM API client functions."""

    def test_get_chat_client_with_api_key(self):
        """Test getting chat client with API key."""
        with patch.dict(os.environ, {"CHAT_API_KEY": "test-key"}):
            client = get_chat_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "test-key"

    def test_get_chat_client_with_base_url(self):
        """Test getting chat client with custom base URL."""
        with patch.dict(
            os.environ,
            {"CHAT_API_KEY": "test-key", "CHAT_API_BASE": "https://custom-api.com/v1"},
        ):
            client = get_chat_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "test-key"
            assert client.base_url == "https://custom-api.com/v1/"

    def test_get_chat_client_without_api_key(self):
        """Test getting chat client without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for chat model"
            ):
                get_chat_client()

    def test_get_embeddings_client_with_api_key(self):
        """Test getting embeddings client with API key."""
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}):
            client = get_embeddings_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "test-key"

    def test_get_embeddings_client_with_base_url(self):
        """Test getting embeddings client with custom base URL."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDINGS_API_KEY": "test-key",
                "EMBEDDINGS_API_BASE": "https://custom-embeddings.com/v1",
            },
        ):
            client = get_embeddings_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "test-key"
            assert client.base_url == "https://custom-embeddings.com/v1/"

    def test_get_embeddings_client_without_api_key(self):
        """Test getting embeddings client without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for embeddings"
            ):
                get_embeddings_client()

    def test_get_chat_fallback_client_with_fallback_key(self):
        """Test getting chat fallback client with fallback API key."""
        with patch.dict(
            os.environ,
            {"CHAT_API_KEY": "primary-key", "CHAT_FALLBACK_API_KEY": "fallback-key"},
        ):
            client = get_chat_fallback_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "fallback-key"

    def test_get_chat_fallback_client_with_inherited_key(self):
        """Test getting chat fallback client with inherited API key."""
        with patch.dict(os.environ, {"CHAT_API_KEY": "primary-key"}, clear=True):
            client = get_chat_fallback_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "primary-key"

    def test_get_chat_fallback_client_without_any_key(self):
        """Test getting chat fallback client without any API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for fallback chat model"
            ):
                get_chat_fallback_client()

    def test_get_embeddings_fallback_client_with_fallback_key(self):
        """Test getting embeddings fallback client with fallback API key."""
        with patch.dict(
            os.environ,
            {
                "EMBEDDINGS_API_KEY": "primary-key",
                "EMBEDDINGS_FALLBACK_API_KEY": "fallback-key",
            },
        ):
            client = get_embeddings_fallback_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "fallback-key"

    def test_get_embeddings_fallback_client_with_inherited_key(self):
        """Test getting embeddings fallback client with inherited API key."""
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "primary-key"}, clear=True):
            client = get_embeddings_fallback_client()
            assert isinstance(client, openai.OpenAI)
            assert client.api_key == "primary-key"

    def test_get_embeddings_fallback_client_without_any_key(self):
        """Test getting embeddings fallback client without any API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for fallback embeddings"
            ):
                get_embeddings_fallback_client()

    def test_get_adaptive_chat_client_with_model_preference(self):
        """Test getting adaptive chat client with model preference."""
        with patch.dict(os.environ, {"CHAT_API_KEY": "test-key"}):
            with patch("src.clients.llm_api_client.get_chat_client") as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client

                client, model_used, is_fallback = get_adaptive_chat_client("gpt-4")
                assert client == mock_client
                assert model_used == "gpt-4"
                assert is_fallback is False

    def test_get_adaptive_chat_client_with_primary_model(self):
        """Test getting adaptive chat client with primary model from environment."""
        with patch.dict(
            os.environ, {"CHAT_API_KEY": "test-key", "CHAT_MODEL": "gpt-4-turbo"}
        ):
            with patch("src.clients.llm_api_client.get_chat_client") as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client

                client, model_used, is_fallback = get_adaptive_chat_client()
                assert client == mock_client
                assert model_used == "gpt-4-turbo"
                assert is_fallback is False

    def test_get_adaptive_chat_client_with_fallback_model(self):
        """Test getting adaptive chat client with fallback model."""
        with patch.dict(
            os.environ,
            {
                "CHAT_FALLBACK_API_KEY": "fallback-key",
                "CHAT_FALLBACK_MODEL": "gpt-3.5-turbo",
            },
            clear=True,
        ):
            with patch(
                "src.clients.llm_api_client.get_chat_fallback_client"
            ) as mock_get_client:
                mock_client = Mock()
                mock_get_client.return_value = mock_client

                client, model_used, is_fallback = get_adaptive_chat_client()
                assert client == mock_client
                assert model_used == "gpt-3.5-turbo"
                assert is_fallback is True

    def test_get_adaptive_chat_client_without_any_config(self):
        """Test getting adaptive chat client without any configuration raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No valid API configuration available"
            ):
                get_adaptive_chat_client()

    def test_validate_chat_config_with_valid_config(self):
        """Test validating chat configuration with valid config."""
        with patch.dict(
            os.environ, {"CHAT_API_KEY": "test-key", "CHAT_MODEL": "gpt-4"}
        ):
            result = validate_chat_config()
            assert result is True

    def test_validate_chat_config_without_api_key(self):
        """Test validating chat configuration without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for chat model"
            ):
                validate_chat_config()

    def test_validate_chat_fallback_config_with_valid_config(self):
        """Test validating chat fallback configuration with valid config."""
        with patch.dict(os.environ, {"CHAT_API_KEY": "test-key"}):
            result = validate_chat_fallback_config()
            assert result is True

    def test_validate_chat_fallback_config_without_any_key(self):
        """Test validating chat fallback configuration without any key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for chat fallback"
            ):
                validate_chat_fallback_config()

    def test_validate_embeddings_fallback_config_with_valid_config(self):
        """Test validating embeddings fallback configuration with valid config."""
        with patch.dict(os.environ, {"EMBEDDINGS_API_KEY": "test-key"}):
            result = validate_embeddings_fallback_config()
            assert result is True

    def test_validate_embeddings_fallback_config_without_any_key(self):
        """Test validating embeddings fallback configuration without any key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No API key configured for embeddings fallback"
            ):
                validate_embeddings_fallback_config()

    def test_get_effective_fallback_config(self):
        """Test getting effective fallback configuration."""
        with patch.dict(
            os.environ,
            {
                "CHAT_API_KEY": "primary-key",
                "CHAT_FALLBACK_API_KEY": "fallback-key",
                "CHAT_FALLBACK_MODEL": "gpt-3.5-turbo",
                "CHAT_FALLBACK_API_BASE": "https://fallback-api.com/v1",
                "EMBEDDINGS_API_KEY": "embeddings-key",
                "EMBEDDINGS_FALLBACK_MODEL": "text-embedding-3-large",
            },
            clear=True,
        ):
            config = get_effective_fallback_config()

            assert config["chat_fallback"]["model"] == "gpt-3.5-turbo"
            assert config["chat_fallback"]["api_key_source"] == "CHAT_FALLBACK_API_KEY"
            assert config["chat_fallback"]["base_url"] == "https://fallback-api.com/v1"
            assert (
                config["chat_fallback"]["base_url_source"] == "CHAT_FALLBACK_API_BASE"
            )

            assert config["embeddings_fallback"]["model"] == "text-embedding-3-large"
            assert (
                config["embeddings_fallback"]["api_key_source"]
                == "EMBEDDINGS_API_KEY (inherited)"
            )
