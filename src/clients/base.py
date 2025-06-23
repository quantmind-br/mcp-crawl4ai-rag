"""
Base client interfaces and abstractions.

This module defines the base classes and interfaces for all client implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import openai


class BaseClient(ABC):
    """Base abstract class for all client implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the client and clean up resources."""
        pass


class BaseOpenAIClient(BaseClient):
    """Base class for OpenAI-compatible clients with fallback support."""
    
    def __init__(self):
        self.primary_client: Optional[openai.OpenAI] = None
        self.fallback_client: Optional[openai.OpenAI] = None
        self.use_fallback: bool = False
    
    @abstractmethod
    def get_primary_config(self) -> Tuple[str, str, Optional[str], Any]:
        """
        Get primary model configuration.
        
        Returns:
            Tuple of (model_name, api_key, api_base, additional_params)
        """
        pass
    
    @abstractmethod
    def get_fallback_config(self) -> Tuple[Optional[str], Optional[str], Optional[str], Any]:
        """
        Get fallback model configuration.
        
        Returns:
            Tuple of (model_name, api_key, api_base, additional_params)
        """
        pass
    
    @abstractmethod
    def get_use_fallback_flag(self) -> bool:
        """Check if fallback is enabled."""
        pass
    
    async def initialize(self) -> None:
        """Initialize primary and fallback clients."""
        # Initialization is handled lazily in get_client methods
        pass
    
    async def close(self) -> None:
        """Close clients."""
        # OpenAI clients don't need explicit closing
        pass