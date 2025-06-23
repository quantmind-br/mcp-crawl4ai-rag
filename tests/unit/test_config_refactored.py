"""
Unit tests for the configuration module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import Config


class TestConfig:
    """Test suite for the Config class."""
    
    def test_config_initialization(self):
        """Test that config initializes properly."""
        config = Config()
        assert config is not None
    
    @patch.dict(os.environ, {}, clear=True)
    def test_default_values(self):
        """Test that default values are returned when environment variables are not set."""
        config = Config()
        
        # Server defaults
        assert config.HOST == "0.0.0.0"
        assert config.PORT == "8051"
        assert config.TRANSPORT == "sse"
        
        # Model defaults
        assert config.CHAT_MODEL == "gpt-4o-mini"
        assert config.EMBEDDING_MODEL == "text-embedding-3-small"
        assert config.RERANKING_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Performance defaults
        assert config.MAX_CRAWL_DEPTH == 3
        assert config.MAX_CONCURRENT_CRAWLS == 10
        assert config.CHUNK_SIZE == 5000
        assert config.DEFAULT_MATCH_COUNT == 5
        
        # Feature flags defaults
        assert config.USE_KNOWLEDGE_GRAPH is False
        assert config.USE_RERANKING is False
        assert config.USE_HYBRID_SEARCH is False
        assert config.USE_AGENTIC_RAG is False
        assert config.USE_CHAT_MODEL_FALLBACK is False
        assert config.USE_EMBEDDING_MODEL_FALLBACK is False
    
    @patch.dict(os.environ, {
        "HOST": "127.0.0.1",
        "PORT": "9000",
        "TRANSPORT": "stdio",
        "CHAT_MODEL": "gpt-4",
        "USE_KNOWLEDGE_GRAPH": "true",
        "MAX_CRAWL_DEPTH": "5",
        "RATE_LIMIT_DELAY": "1.0"
    })
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        config = Config()
        
        assert config.HOST == "127.0.0.1"
        assert config.PORT == "9000"
        assert config.TRANSPORT == "stdio"
        assert config.CHAT_MODEL == "gpt-4"
        assert config.USE_KNOWLEDGE_GRAPH is True
        assert config.MAX_CRAWL_DEPTH == 5
        assert config.RATE_LIMIT_DELAY == 1.0
    
    @patch.dict(os.environ, {
        "USE_KNOWLEDGE_GRAPH": "false",
        "USE_RERANKING": "true",
        "USE_HYBRID_SEARCH": "TRUE",
        "USE_AGENTIC_RAG": "False"
    })
    def test_boolean_flag_parsing(self):
        """Test that boolean flags are parsed correctly regardless of case."""
        config = Config()
        
        assert config.USE_KNOWLEDGE_GRAPH is False
        assert config.USE_RERANKING is True
        assert config.USE_HYBRID_SEARCH is True
        assert config.USE_AGENTIC_RAG is False
    
    @patch.dict(os.environ, {
        "MAX_CRAWL_DEPTH": "invalid",
        "RATE_LIMIT_DELAY": "not_a_number"
    })
    def test_invalid_numeric_values(self):
        """Test that invalid numeric values raise appropriate errors."""
        config = Config()
        
        with pytest.raises(ValueError):
            _ = config.MAX_CRAWL_DEPTH
        
        with pytest.raises(ValueError):
            _ = config.RATE_LIMIT_DELAY
    
    @patch('src.config.Path.exists')
    def test_is_containerized_detection(self, mock_exists):
        """Test containerized environment detection."""
        config = Config()
        
        # Test Docker detection
        mock_exists.return_value = True
        assert config._is_containerized() is True
        
        # Test without Docker
        mock_exists.return_value = False
        with patch.dict(os.environ, {"CONTAINER": "true"}):
            assert config._is_containerized() is True
        
        with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
            assert config._is_containerized() is True
        
        with patch.dict(os.environ, {}, clear=True):
            assert config._is_containerized() is False
    
    @patch('src.config.Config._load_dotenv_file')
    @patch('src.config.Config._is_containerized')
    def test_environment_loading(self, mock_is_containerized, mock_load_dotenv):
        """Test environment loading logic."""
        # Test containerized environment (should not load .env)
        mock_is_containerized.return_value = True
        config = Config()
        mock_load_dotenv.assert_not_called()
        
        # Test non-containerized environment (should load .env)
        mock_is_containerized.return_value = False
        config = Config()
        mock_load_dotenv.assert_called_once()
    
    @patch('src.config.load_dotenv')
    @patch('src.config.Path.exists')
    def test_dotenv_loading(self, mock_exists, mock_load_dotenv):
        """Test .env file loading."""
        config = Config()
        
        # Test when .env exists
        mock_exists.return_value = True
        config._load_dotenv_file()
        mock_load_dotenv.assert_called_once()
        
        # Test when .env doesn't exist
        mock_exists.return_value = False
        mock_load_dotenv.reset_mock()
        config._load_dotenv_file()
        mock_load_dotenv.assert_not_called()
    
    def test_optional_environment_variables(self):
        """Test that optional environment variables return None when not set."""
        config = Config()
        
        assert config.SUPABASE_URL is None
        assert config.SUPABASE_SERVICE_KEY is None
        assert config.NEO4J_URI is None
        assert config.NEO4J_USER is None
        assert config.NEO4J_PASSWORD is None
        assert config.EMBEDDING_MODEL_API_BASE is None