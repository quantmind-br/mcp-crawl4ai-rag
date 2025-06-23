"""
Base tool interface for MCP tools.

This module defines the base interface and common functionality for all MCP tools.
"""

from abc import ABC, abstractmethod
from typing import Any
from mcp.server.fastmcp import Context


class BaseTool(ABC):
    """Base abstract class for MCP tools."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass
    
    @abstractmethod
    async def execute(self, ctx: Context, **kwargs) -> Any:
        """Execute the tool."""
        pass