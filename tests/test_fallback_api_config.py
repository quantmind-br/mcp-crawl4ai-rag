"""
# ruff: noqa: E402
Integration tests for fallback API configuration system.

Tests the fallback API configuration with inheritance patterns,
mixed provider scenarios, and adaptive client functionality.
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


class TestFallbackAPIConfiguration:
    """Test the fallback API configuration system with inheritance."""

    def setup_method(self):
        """Set up test environment for each test."""
        # Save original environment (including conftest values)
        self.original_env = {
            key: os.environ.get(key)
            for key in [
                "CHAT_MODEL",
                "CHAT_API_KEY",
                "CHAT_API_BASE",
                "CHAT_FALLBACK_MODEL",
                "CHAT_FALLBACK_API_KEY",
                "CHAT_FALLBACK_API_BASE",
                "EMBEDDINGS_MODEL",
                "EMBEDDINGS_API_KEY",
                "EMBEDDINGS_API_BASE",
                "EMBEDDINGS_FALLBACK_MODEL",
                "EMBEDDINGS_FALLBACK_API_KEY",
                "EMBEDDINGS_FALLBACK_API_BASE",
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

    def test_chat_fallback_api_key_inheritance(self):
        """Test that chat fallback inherits primary API key when not specified."""
        # Set primary configuration only
        os.environ["CHAT_API_KEY"] = "primary-chat-key"
        os.environ["CHAT_API_BASE"] = "https://primary.api.com/v1"

        from src.clients.llm_api_client import get_chat_fallback_client

        client = get_chat_fallback_client()
        assert client.api_key == "primary-chat-key"
        assert str(client.base_url).rstrip("/") == "https://primary.api.com/v1"

    def test_chat_fallback_api_key_override(self):
        """Test that explicit fallback API key overrides inheritance."""
        # Set both primary and fallback configuration
        os.environ["CHAT_API_KEY"] = "primary-chat-key"
        os.environ["CHAT_API_BASE"] = "https://primary.api.com/v1"
        os.environ["CHAT_FALLBACK_API_KEY"] = "fallback-chat-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = "https://fallback.api.com/v1"

        from src.clients.llm_api_client import get_chat_fallback_client

        client = get_chat_fallback_client()
        assert client.api_key == "fallback-chat-key"
        assert str(client.base_url).rstrip("/") == "https://fallback.api.com/v1"

    def test_embeddings_fallback_api_key_inheritance(self):
        """Test that embeddings fallback inherits primary API key when not specified."""
        # Set primary configuration only
        os.environ["EMBEDDINGS_API_KEY"] = "primary-embeddings-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://primary.api.com/v1"

        from src.clients.llm_api_client import get_embeddings_fallback_client

        client = get_embeddings_fallback_client()
        assert client.api_key == "primary-embeddings-key"
        assert str(client.base_url).rstrip("/") == "https://primary.api.com/v1"

    def test_embeddings_fallback_api_key_override(self):
        """Test that explicit fallback API key overrides inheritance."""
        # Set both primary and fallback configuration
        os.environ["EMBEDDINGS_API_KEY"] = "primary-embeddings-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://primary.api.com/v1"
        os.environ["EMBEDDINGS_FALLBACK_API_KEY"] = "fallback-embeddings-key"
        os.environ["EMBEDDINGS_FALLBACK_API_BASE"] = "https://fallback.api.com/v1"

        from src.clients.llm_api_client import get_embeddings_fallback_client

        client = get_embeddings_fallback_client()
        assert client.api_key == "fallback-embeddings-key"
        assert str(client.base_url).rstrip("/") == "https://fallback.api.com/v1"

    def test_mixed_provider_configuration(self):
        """Test using different providers for primary and fallback."""
        # Primary via OpenRouter, fallback via OpenAI (explicitly configured)
        os.environ["CHAT_API_KEY"] = "openrouter-key"
        os.environ["CHAT_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["CHAT_FALLBACK_API_KEY"] = "openai-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = (
            "https://api.openai.com/v1"  # Explicitly set for different provider
        )

        from src.clients.llm_api_client import get_chat_client, get_chat_fallback_client

        primary_client = get_chat_client()
        fallback_client = get_chat_fallback_client()

        # Primary should use OpenRouter
        assert primary_client.api_key == "openrouter-key"
        assert (
            str(primary_client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
        )

        # Fallback should use OpenAI (explicitly configured)
        assert fallback_client.api_key == "openai-key"
        assert str(fallback_client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_adaptive_chat_client_primary_success(self):
        """Test adaptive client uses primary when available."""
        # Set primary configuration
        os.environ["CHAT_MODEL"] = "gpt-4"
        os.environ["CHAT_API_KEY"] = "primary-key"
        os.environ["CHAT_API_BASE"] = "https://primary.api.com/v1"

        from src.clients.llm_api_client import get_adaptive_chat_client

        client, model_used, is_fallback = get_adaptive_chat_client()

        assert client.api_key == "primary-key"
        assert str(client.base_url).rstrip("/") == "https://primary.api.com/v1"
        assert model_used == "gpt-4"
        assert is_fallback is False

    def test_adaptive_chat_client_fallback_when_no_primary(self):
        """Test adaptive client uses fallback when primary not available."""
        # Set only fallback configuration
        os.environ["CHAT_FALLBACK_MODEL"] = "gpt-3.5-turbo"
        os.environ["CHAT_FALLBACK_API_KEY"] = "fallback-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = "https://fallback.api.com/v1"

        from src.clients.llm_api_client import get_adaptive_chat_client

        client, model_used, is_fallback = get_adaptive_chat_client()

        assert client.api_key == "fallback-key"
        assert str(client.base_url).rstrip("/") == "https://fallback.api.com/v1"
        assert model_used == "gpt-3.5-turbo"
        assert is_fallback is True

    def test_adaptive_chat_client_with_inheritance(self):
        """Test adaptive client fallback with inheritance from primary."""
        # Set primary configuration that fallback will inherit
        os.environ["CHAT_API_KEY"] = "inherited-key"
        os.environ["CHAT_API_BASE"] = "https://inherited.api.com/v1"
        os.environ["CHAT_FALLBACK_MODEL"] = "gpt-3.5-turbo"
        # No primary model set, so should use fallback

        from src.clients.llm_api_client import get_adaptive_chat_client

        client, model_used, is_fallback = get_adaptive_chat_client()

        assert client.api_key == "inherited-key"
        assert str(client.base_url).rstrip("/") == "https://inherited.api.com/v1"
        assert model_used == "gpt-3.5-turbo"
        assert is_fallback is True

    def test_adaptive_chat_client_model_preference(self):
        """Test adaptive client with specific model preference."""
        # Set fallback configuration
        os.environ["CHAT_FALLBACK_API_KEY"] = "fallback-key"

        from src.clients.llm_api_client import get_adaptive_chat_client

        client, model_used, is_fallback = get_adaptive_chat_client(
            model_preference="custom-model"
        )

        assert client.api_key == "fallback-key"
        assert model_used == "custom-model"
        # is_fallback might be True because we're using fallback client

    def test_validation_chat_fallback_config_success(self):
        """Test validation of valid chat fallback configuration."""
        # Set valid configuration
        os.environ["CHAT_API_KEY"] = "valid-key"
        os.environ["CHAT_API_BASE"] = "https://valid.api.com/v1"

        from src.clients.llm_api_client import validate_chat_fallback_config

        result = validate_chat_fallback_config()
        assert result is True

    def test_validation_chat_fallback_config_no_key(self):
        """Test validation fails when no API key is available."""
        # No API keys set

        from src.clients.llm_api_client import validate_chat_fallback_config

        with pytest.raises(ValueError, match="No API key configured for chat fallback"):
            validate_chat_fallback_config()

    def test_validation_embeddings_fallback_config_success(self):
        """Test validation of valid embeddings fallback configuration."""
        # Set valid configuration
        os.environ["EMBEDDINGS_API_KEY"] = "valid-key"
        os.environ["EMBEDDINGS_API_BASE"] = "https://valid.api.com/v1"

        from src.clients.llm_api_client import validate_embeddings_fallback_config

        result = validate_embeddings_fallback_config()
        assert result is True

    def test_validation_embeddings_fallback_config_no_key(self):
        """Test validation fails when no API key is available."""
        # No API keys set

        from src.clients.llm_api_client import validate_embeddings_fallback_config

        with pytest.raises(
            ValueError, match="No API key configured for embeddings fallback"
        ):
            validate_embeddings_fallback_config()

    def test_get_effective_fallback_config(self):
        """Test effective fallback configuration resolution."""
        # Set mixed configuration with inheritance
        os.environ["CHAT_API_KEY"] = "primary-chat-key"
        os.environ["CHAT_FALLBACK_MODEL"] = "gpt-3.5-turbo"
        os.environ["EMBEDDINGS_FALLBACK_API_KEY"] = "explicit-embeddings-key"
        os.environ["EMBEDDINGS_FALLBACK_API_BASE"] = "https://explicit.api.com/v1"

        from src.clients.llm_api_client import get_effective_fallback_config

        config = get_effective_fallback_config()

        # Chat should inherit primary key
        assert config["chat_fallback"]["model"] == "gpt-3.5-turbo"
        assert config["chat_fallback"]["api_key_source"] == "CHAT_API_KEY (inherited)"

        # Embeddings should use explicit fallback configuration
        assert (
            config["embeddings_fallback"]["api_key_source"]
            == "EMBEDDINGS_FALLBACK_API_KEY"
        )
        assert (
            config["embeddings_fallback"]["base_url"] == "https://explicit.api.com/v1"
        )

    def test_no_configuration_error_handling(self):
        """Test error handling when no configuration is available."""
        # No environment variables set

        from src.clients.llm_api_client import get_adaptive_chat_client

        with pytest.raises(ValueError, match="No valid API configuration available"):
            get_adaptive_chat_client()

    @patch("src.utils.get_chat_client")
    @patch("src.utils.get_chat_fallback_client")
    def test_performance_fallback_switching_time(
        self, mock_fallback_client, mock_primary_client
    ):
        """Test that fallback switching is fast."""
        # Mock clients
        mock_primary_client.side_effect = ValueError("Primary failed")
        mock_fallback_client.return_value = Mock()

        # Set fallback configuration
        os.environ["CHAT_FALLBACK_API_KEY"] = "fallback-key"
        os.environ["CHAT_FALLBACK_MODEL"] = "gpt-3.5-turbo"
        os.environ["CHAT_API_KEY"] = "primary-key"  # This will fail
        os.environ["CHAT_MODEL"] = "gpt-4"

        from src.clients.llm_api_client import get_adaptive_chat_client

        # Measure fallback switching time
        start_time = time.time()
        client, model_used, is_fallback = get_adaptive_chat_client()
        end_time = time.time()

        # Should be very fast (under 100ms)
        elapsed = end_time - start_time
        assert elapsed < 0.1, f"Fallback switching too slow: {elapsed:.3f}s"
        assert is_fallback is True
        assert model_used == "gpt-3.5-turbo"

    def test_base_url_validation(self):
        """Test base URL validation in fallback configuration."""
        # Set invalid base URL
        os.environ["CHAT_API_KEY"] = "valid-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = "not-a-valid-url"

        from src.clients.llm_api_client import validate_chat_fallback_config

        with pytest.raises(ValueError, match="Invalid"):
            validate_chat_fallback_config()

    def test_partial_inheritance_scenarios(self):
        """Test various partial inheritance scenarios."""
        # Scenario 1: Inherit key but override base URL
        os.environ["CHAT_API_KEY"] = "inherited-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = "https://fallback.api.com/v1"

        from src.clients.llm_api_client import get_chat_fallback_client

        client = get_chat_fallback_client()
        assert client.api_key == "inherited-key"
        assert str(client.base_url).rstrip("/") == "https://fallback.api.com/v1"

    def test_real_world_provider_combinations(self):
        """Test realistic provider combination scenarios."""
        # Scenario: Expensive primary (GPT-4 via OpenAI), cheap fallback (GPT-3.5 via OpenRouter)
        os.environ["CHAT_MODEL"] = "gpt-4"
        os.environ["CHAT_API_KEY"] = "openai-key"
        # Primary uses default OpenAI

        os.environ["CHAT_FALLBACK_MODEL"] = "openai/gpt-3.5-turbo"
        os.environ["CHAT_FALLBACK_API_KEY"] = "openrouter-key"
        os.environ["CHAT_FALLBACK_API_BASE"] = "https://openrouter.ai/api/v1"

        from src.clients.llm_api_client import get_chat_client, get_chat_fallback_client

        primary_client = get_chat_client()
        fallback_client = get_chat_fallback_client()

        # Primary: OpenAI with expensive model
        assert primary_client.api_key == "openai-key"
        assert str(primary_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        # Fallback: OpenRouter with cheaper model
        assert fallback_client.api_key == "openrouter-key"
        assert (
            str(fallback_client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
        )
