"""Core interfaces for knowledge graph components.

This module defines abstract base classes and protocols that all concrete
implementations must follow to ensure consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path

from .models import ParseResult, ParsedClass, ParsedFunction


class LanguageParser(ABC):
    """
    Abstract base class for all language parsers in the Tree-sitter integration.

    This class defines the interface that all language-specific parsers must implement
    to ensure consistent behavior and data structure compatibility with the existing
    Neo4j schema and AST-based parsing output.

    Key Requirements:
    - Output must match existing AST parser structure exactly
    - Must handle parsing errors gracefully
    - Should support extraction of classes, methods, functions, and imports
    - Must maintain backward compatibility with existing data consumers
    """

    def __init__(self, language_name: str):
        """
        Initialize the language parser.

        Args:
            language_name: Name of the programming language (e.g., 'python', 'typescript')
        """
        self.language_name = language_name
        self.supported_extensions: List[str] = []

    @abstractmethod
    def parse(self, file_content: str, file_path: str) -> ParseResult:
        """
        Parse source code and extract structural information.

        This is the main parsing method that must be implemented by all language parsers.
        The output structure must exactly match the format expected by existing Neo4j
        population logic and downstream consumers.

        Args:
            file_content: The source code content as a string
            file_path: Path to the source file (for error reporting and module naming)

        Returns:
            ParseResult containing extracted classes, functions, imports, etc.

        Raises:
            Should not raise exceptions - all errors should be captured in ParseResult.errors
        """
        pass

    @abstractmethod
    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this parser supports the file, False otherwise
        """
        pass

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of file extensions this parser supports.

        Returns:
            List of file extensions including the dot (e.g., ['.py', '.pyi'])
        """
        return self.supported_extensions.copy()

    def get_language_name(self) -> str:
        """
        Get the name of the programming language this parser handles.

        Returns:
            Language name (e.g., 'python', 'typescript', 'java')
        """
        return self.language_name

    def _convert_to_dict_structure(
        self, parsed_classes: List[ParsedClass], parsed_functions: List[ParsedFunction]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert parsed dataclasses to the dictionary structure expected by Neo4j.

        This helper method ensures consistent conversion from the structured dataclasses
        to the exact dictionary format that existing Neo4j population code expects.

        Args:
            parsed_classes: List of ParsedClass objects
            parsed_functions: List of ParsedFunction objects

        Returns:
            Tuple of (classes_dict_list, functions_dict_list) matching existing format
        """
        classes_dict = []
        for cls in parsed_classes:
            methods_dict = []
            for method in cls.methods:
                methods_dict.append(
                    {
                        "name": method.name,
                        "params": method.params,
                        "return_type": method.return_type,
                        "line_start": method.line_start,
                        "line_end": method.line_end,
                    }
                )

            attributes_dict = []
            for attr in cls.attributes:
                attributes_dict.append(
                    {
                        "name": attr.name,
                        "type": attr.type,
                        "line_start": attr.line_start,
                        "line_end": attr.line_end,
                    }
                )

            classes_dict.append(
                {
                    "name": cls.name,
                    "full_name": cls.full_name,
                    "methods": methods_dict,
                    "attributes": attributes_dict,
                    "line_start": cls.line_start,
                    "line_end": cls.line_end,
                }
            )

        functions_dict = []
        for func in parsed_functions:
            functions_dict.append(
                {
                    "name": func.name,
                    "full_name": func.full_name,
                    "params": func.params,
                    "return_type": func.return_type,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                }
            )

        return classes_dict, functions_dict

    def _safe_parse_with_fallback(
        self, file_content: str, file_path: str
    ) -> ParseResult:
        """
        Template method for safe parsing with error handling.

        This method provides a common pattern for parsing with graceful error handling
        and fallback to empty results when parsing fails completely.

        Args:
            file_content: Source code to parse
            file_path: File path for error reporting

        Returns:
            ParseResult with either parsed data or empty fallback structure
        """
        try:
            return self.parse(file_content, file_path)
        except Exception as e:
            # Create fallback empty result structure
            return ParseResult(
                module_name="unknown",
                file_path=file_path,
                classes=[],
                functions=[],
                imports=[],
                line_count=len(file_content.splitlines()),
                language=self.language_name,
                errors=[f"Parser failed: {str(e)}"],
            )

    def _extract_module_name(self, file_path: str) -> str:
        """
        Extract module name from file path using standard conventions.

        Args:
            file_path: Path to the source file

        Returns:
            Module name suitable for Neo4j storage
        """
        path = Path(file_path)

        # Remove file extension and use stem as module name
        module_name = path.stem

        # For files like __init__.py, use parent directory name
        if module_name == "__init__":
            module_name = path.parent.name

        return module_name
