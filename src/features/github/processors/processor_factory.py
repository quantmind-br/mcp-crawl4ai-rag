"""
Processor factory and registration system.

This module provides a factory pattern for creating file processors
and a registry system for managing different processor types.
"""

import logging
from typing import Dict, List, Optional, Type

from .base_processor import IFileProcessor
from .markdown_processor import MarkdownProcessor
from .python_processor import PythonProcessor
from .typescript_processor import TypeScriptProcessor
from .config_processor import ConfigProcessor
from ..core.exceptions import ProcessorNotFoundError, ConfigurationError
from ..config.settings import PROCESSOR_PRIORITY, get_language_for_extension


class ProcessorRegistry:
    """Registry for managing file processors."""

    def __init__(self):
        """Initialize processor registry."""
        self._processors: Dict[str, Type[IFileProcessor]] = {}
        self._instances: Dict[str, IFileProcessor] = {}
        self.logger = logging.getLogger(__name__)

    def register(self, name: str, processor_class: Type[IFileProcessor]) -> None:
        """
        Register a processor class.

        Args:
            name: Processor name
            processor_class: Processor class to register

        Raises:
            ConfigurationError: If processor is already registered
        """
        if name in self._processors:
            raise ConfigurationError(
                f"Processor '{name}' is already registered",
                config_key="processor_registry",
                config_value=name,
            )

        self._processors[name] = processor_class
        self.logger.debug(f"Registered processor: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a processor.

        Args:
            name: Processor name to unregister
        """
        if name in self._processors:
            del self._processors[name]
        if name in self._instances:
            del self._instances[name]
        self.logger.debug(f"Unregistered processor: {name}")

    def get_processor(self, name: str) -> IFileProcessor:
        """
        Get processor instance by name.

        Args:
            name: Processor name

        Returns:
            Processor instance

        Raises:
            ProcessorNotFoundError: If processor is not registered
        """
        if name not in self._processors:
            raise ProcessorNotFoundError(
                f"No processor registered with name '{name}'", file_extension=None
            )

        # Use cached instance or create new one
        if name not in self._instances:
            processor_class = self._processors[name]
            self._instances[name] = processor_class()

        return self._instances[name]

    def get_processor_for_file(self, file_path: str) -> Optional[IFileProcessor]:
        """
        Get the best processor for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Processor instance or None if no suitable processor found
        """
        import os

        file_ext = os.path.splitext(file_path)[1].lower()

        if not file_ext:
            return None

        # Find processors that can handle this file
        suitable_processors = []
        for name, processor_class in self._processors.items():
            # Get processor instance to check if it can process the file
            processor = self.get_processor(name)
            if processor.can_process(file_path):
                suitable_processors.append((name, processor))

        if not suitable_processors:
            return None

        # If multiple processors can handle the file, choose by priority
        if len(suitable_processors) == 1:
            return suitable_processors[0][1]

        # Sort by processor priority (higher priority first)
        language = get_language_for_extension(file_ext)
        PROCESSOR_PRIORITY.get(language, 0)

        # Choose highest priority processor
        best_processor = max(
            suitable_processors, key=lambda x: PROCESSOR_PRIORITY.get(x[0], 0)
        )

        return best_processor[1]

    def list_processors(self) -> List[str]:
        """
        List all registered processor names.

        Returns:
            List of processor names
        """
        return list(self._processors.keys())

    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions across all processors.

        Returns:
            List of supported file extensions
        """
        extensions = set()
        for name in self._processors:
            processor = self.get_processor(name)
            extensions.update(processor.get_supported_extensions())
        return sorted(list(extensions))

    def clear(self) -> None:
        """Clear all registered processors."""
        self._processors.clear()
        self._instances.clear()
        self.logger.debug("Cleared all processors from registry")


class ProcessorFactory:
    """Factory for creating and managing file processors."""

    def __init__(self, registry: ProcessorRegistry = None):
        """
        Initialize processor factory.

        Args:
            registry: Processor registry instance (creates default if None)
        """
        self.registry = registry or self._create_default_registry()
        self.logger = logging.getLogger(__name__)

    def _create_default_registry(self) -> ProcessorRegistry:
        """Create a registry with default processors."""
        registry = ProcessorRegistry()

        # Register default processors
        registry.register("markdown", MarkdownProcessor)
        registry.register("python", PythonProcessor)
        registry.register("typescript", TypeScriptProcessor)
        registry.register("config", ConfigProcessor)

        return registry

    def get_processor_for_file(self, file_path: str) -> Optional[IFileProcessor]:
        """
        Get the appropriate processor for a file.

        Args:
            file_path: Path to the file

        Returns:
            Processor instance or None if no suitable processor
        """
        return self.registry.get_processor_for_file(file_path)

    def get_processor_by_name(self, name: str) -> IFileProcessor:
        """
        Get processor by name.

        Args:
            name: Processor name

        Returns:
            Processor instance

        Raises:
            ProcessorNotFoundError: If processor not found
        """
        return self.registry.get_processor(name)

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        return self.registry.get_supported_extensions()

    def register_processor(
        self, name: str, processor_class: Type[IFileProcessor]
    ) -> None:
        """
        Register a new processor.

        Args:
            name: Processor name
            processor_class: Processor class
        """
        self.registry.register(name, processor_class)

    def list_processors(self) -> List[str]:
        """List all available processors."""
        return self.registry.list_processors()


# Global default factory instance
_default_factory: Optional[ProcessorFactory] = None


def get_default_factory() -> ProcessorFactory:
    """
    Get the default processor factory instance.

    Returns:
        Default ProcessorFactory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = ProcessorFactory()
    return _default_factory


def create_processor_for_file(file_path: str) -> Optional[IFileProcessor]:
    """
    Create a processor for a specific file using the default factory.

    Args:
        file_path: Path to the file

    Returns:
        Processor instance or None if no suitable processor
    """
    factory = get_default_factory()
    return factory.get_processor_for_file(file_path)


def get_supported_extensions() -> List[str]:
    """
    Get all supported file extensions from the default factory.

    Returns:
        List of supported file extensions
    """
    factory = get_default_factory()
    return factory.get_supported_extensions()
