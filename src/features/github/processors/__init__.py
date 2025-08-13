"""
File processors for different file types.

This package provides processing capabilities for various file types
including markdown, Python, TypeScript, configuration files, and more.
"""

from .base_processor import IFileProcessor, BaseFileProcessor
from .markdown_processor import MarkdownProcessor
from .mdx_processor import MDXProcessor
from .python_processor import PythonProcessor
from .typescript_processor import TypeScriptProcessor
from .config_processor import ConfigProcessor
from .processor_factory import ProcessorFactory, ProcessorRegistry

__all__ = [
    "IFileProcessor",
    "BaseFileProcessor",
    "MarkdownProcessor",
    "MDXProcessor",
    "PythonProcessor",
    "TypeScriptProcessor",
    "ConfigProcessor",
    "ProcessorFactory",
    "ProcessorRegistry",
]
