"""
Parser Factory

Central factory class for creating and managing language parsers in the Tree-sitter
multi-language code analysis integration. Handles parser selection, caching, and
language detection based on file extensions and content analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Any
from functools import lru_cache

from ..core.interfaces import LanguageParser

# Configure logging
logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory class for creating and managing language parsers.

    This factory implements the Factory pattern to provide a centralized way to:
    - Detect programming languages from file extensions
    - Create appropriate parser instances
    - Cache parsers for performance
    - Handle unsupported languages gracefully
    - Manage parser lifecycle

    The factory maintains backward compatibility with existing AST-based parsing
    while providing the foundation for multi-language Tree-sitter integration.
    """

    def __init__(self):
        """Initialize the parser factory with language mappings and cache."""

        # File extension to language mapping
        self.extension_mapping: Dict[str, str] = {
            # Python
            ".py": "python",
            ".pyi": "python",
            # JavaScript/TypeScript
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            # Java
            ".java": "java",
            # Go
            ".go": "go",
            # Rust
            ".rs": "rust",
            # C/C++
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".cc": "cpp",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".hh": "cpp",
            # C#
            ".cs": "c_sharp",
            # PHP
            ".php": "php",
            ".php3": "php",
            ".php4": "php",
            ".php5": "php",
            ".phtml": "php",
            # Ruby
            ".rb": "ruby",
            ".rbw": "ruby",
            # Kotlin
            ".kt": "kotlin",
            ".kts": "kotlin",
        }

        # Cache for parser instances (LRU cache with max 50 parsers)
        self._parser_cache: Dict[str, LanguageParser] = {}
        self._cache_max_size = 50

        # Registry of available parser classes
        self._parser_registry: Dict[str, type] = {}

        # Set of supported languages
        self.supported_languages: Set[str] = set(self.extension_mapping.values())

        # Statistics for monitoring
        self.stats = {
            "parsers_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "unknown_languages": 0,
            "successful_detections": 0,
        }

        logger.info(
            f"ParserFactory initialized with support for {len(self.supported_languages)} languages"
        )

    def register_parser(self, language: str, parser_class: type) -> None:
        """
        Register a parser class for a specific language.

        Args:
            language: The programming language name
            parser_class: The LanguageParser subclass to register
        """
        if not issubclass(parser_class, LanguageParser):
            raise ValueError("Parser class must be a subclass of LanguageParser")

        self._parser_registry[language] = parser_class
        self.supported_languages.add(language)

        logger.debug(f"Registered parser for language: {language}")

    def detect_language(self, file_path: str, content: Optional[str] = None) -> str:
        """
        Detect programming language from file path and optionally file content.

        Args:
            file_path: Path to the source file
            content: Optional file content for content-based detection

        Returns:
            Detected language name or 'unknown' if not supported
        """
        # Primary detection: file extension
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in self.extension_mapping:
            language = self.extension_mapping[extension]
            self.stats["successful_detections"] += 1
            logger.debug(
                f"Detected language '{language}' from extension '{extension}' for {file_path}"
            )
            return language

        # Secondary detection: special filename patterns
        filename = path.name.lower()

        # Special cases for common files without extensions
        special_files = {
            "makefile": "c",
            "dockerfile": "dockerfile",
            "rakefile": "ruby",
            "gemfile": "ruby",
            "cargo.toml": "rust",
            "go.mod": "go",
            "go.sum": "go",
        }

        if filename in special_files:
            language = special_files[filename]
            self.stats["successful_detections"] += 1
            logger.debug(f"Detected language '{language}' from filename '{filename}'")
            return language

        # Tertiary detection: content-based analysis (if content provided)
        if content:
            content_language = self._detect_language_from_content(content)
            if content_language != "unknown":
                self.stats["successful_detections"] += 1
                logger.debug(
                    f"Detected language '{content_language}' from content analysis"
                )
                return content_language

        # Language detection failed
        self.stats["unknown_languages"] += 1
        logger.warning(f"Could not detect language for file: {file_path}")
        return None

    def _detect_language_from_content(self, content: str) -> str:
        """
        Detect language from file content using heuristics.

        Args:
            content: The file content to analyze

        Returns:
            Detected language name or 'unknown'
        """
        content_lower = content.lower()
        lines = content.split("\n")

        # Check for shebangs
        if lines and lines[0].startswith("#!"):
            shebang = lines[0].lower()
            if "python" in shebang:
                return "python"
            elif "node" in shebang or "js" in shebang:
                return "javascript"
            elif "ruby" in shebang:
                return "ruby"
            elif "php" in shebang:
                return "php"

        # Language-specific patterns
        patterns = {
            "python": ["def ", "import ", "from ", "class ", "__name__", "__main__"],
            "javascript": [
                "function ",
                "var ",
                "let ",
                "const ",
                "=>",
                "require(",
                "module.exports",
            ],
            "typescript": [
                "interface ",
                "type ",
                ": string",
                ": number",
                ": boolean",
                "export ",
                "import ",
            ],
            "java": [
                "public class",
                "public static void main",
                "import java.",
                "package ",
            ],
            "go": ["package ", "func ", 'import "', "var ", "type ", "interface"],
            "rust": ["fn ", "let ", "mut ", "use ", "mod ", "impl ", "trait "],
            "c": ["#include", "int main", "printf(", "malloc(", "struct "],
            "cpp": ["#include", "using namespace", "std::", "class ", "::"],
            "c_sharp": [
                "using System",
                "namespace ",
                "public class",
                "static void Main",
            ],
            "php": ["<?php", "$_", "function ", "class ", "echo "],
            "ruby": ["def ", "class ", "require ", "puts ", "end"],
            "kotlin": ["fun ", "val ", "var ", "class ", "package "],
        }

        # Score each language based on pattern matches
        scores = {}
        for language, pattern_list in patterns.items():
            score = sum(1 for pattern in pattern_list if pattern in content_lower)
            if score > 0:
                scores[language] = score

        if scores:
            # Return language with highest score
            best_language = max(scores.items(), key=lambda x: x[1])[0]
            return best_language

        return "unknown"

    def get_parser(self, language: str) -> Optional[LanguageParser]:
        """
        Get a parser instance for the specified language.

        Args:
            language: The programming language name

        Returns:
            LanguageParser instance or None if language not supported
        """
        if language == "unknown" or language not in self.supported_languages:
            logger.debug(f"No parser available for language: {language}")
            return None

        # Check cache first
        if language in self._parser_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Parser cache hit for language: {language}")
            return self._parser_cache[language]

        # Cache miss - create new parser
        self.stats["cache_misses"] += 1

        # Check if we have a registered parser class
        if language in self._parser_registry:
            parser_class = self._parser_registry[language]
            parser = parser_class(language)

            # Add to cache (with size management)
            self._cache_parser(language, parser)

            self.stats["parsers_created"] += 1
            logger.debug(f"Created new parser for language: {language}")
            return parser

        # No parser class registered - try TreeSitterParser as default, with fallback
        try:
            # Import here to avoid circular imports
            from .tree_sitter_parser import TreeSitterParser

            parser = TreeSitterParser(language)

            # Add to cache
            self._cache_parser(language, parser)

            self.stats["parsers_created"] += 1
            logger.debug(f"Created default TreeSitter parser for language: {language}")
            return parser

        except ImportError as e:
            logger.error(f"Could not import TreeSitterParser: {e}")
            # Fallback to simple heuristic parser to maintain functionality when Tree-sitter is unavailable
            try:
                from .simple_fallback_parser import SimpleFallbackParser

                parser = SimpleFallbackParser(language)
                self._cache_parser(language, parser)
                self.stats["parsers_created"] += 1
                logger.debug(
                    f"Using SimpleFallbackParser for language: {language} (Tree-sitter unavailable)"
                )
                return parser
            except Exception as fe:
                logger.error(f"Failed to create fallback parser for {language}: {fe}")
                return None
        except Exception as e:
            logger.error(f"Failed to create parser for language {language}: {e}")
            return None

    def _cache_parser(self, language: str, parser: LanguageParser) -> None:
        """
        Add parser to cache with size management.

        Args:
            language: The programming language name
            parser: The parser instance to cache
        """
        # If cache is full, remove oldest entry (simple FIFO)
        if len(self._parser_cache) >= self._cache_max_size:
            # Remove the first item (oldest)
            oldest_language = next(iter(self._parser_cache))
            del self._parser_cache[oldest_language]
            logger.debug(f"Evicted parser from cache: {oldest_language}")

        self._parser_cache[language] = parser

    def get_parser_for_file(
        self, file_path: str, content: Optional[str] = None
    ) -> Optional[LanguageParser]:
        """
        Get appropriate parser for a specific file.

        Args:
            file_path: Path to the source file
            content: Optional file content for better detection

        Returns:
            LanguageParser instance or None if not supported
        """
        language = self.detect_language(file_path, content)
        return self.get_parser(language)

    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if a file is supported by any available parser.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is supported, False otherwise
        """
        language = self.detect_language(file_path)
        return language != "unknown"

    def get_supported_extensions(self) -> Set[str]:
        """
        Get set of all supported file extensions.

        Returns:
            Set of file extensions including the dot (e.g., {'.py', '.js'})
        """
        return set(self.extension_mapping.keys())

    def get_supported_languages(self) -> Set[str]:
        """
        Get set of all supported programming languages.

        Returns:
            Set of language names
        """
        return set(self.supported_languages)

    def clear_cache(self) -> None:
        """Clear the parser cache."""
        self._parser_cache.clear()
        logger.debug("Parser cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        Get factory usage statistics.

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the parser cache.

        Returns:
            Dictionary with cache information
        """
        return {
            "cache_size": len(self._parser_cache),
            "max_cache_size": self._cache_max_size,
            "cached_languages": list(self._parser_cache.keys()),
            "hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            ),
        }


# Global factory instance for convenience
_global_factory: Optional[ParserFactory] = None


def get_global_factory() -> ParserFactory:
    """
    Get the global parser factory instance.

    Returns:
        Global ParserFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = ParserFactory()
    return _global_factory


@lru_cache(maxsize=1000)
def detect_language_cached(file_path: str) -> str:
    """
    Cached version of language detection for frequently accessed files.

    Args:
        file_path: Path to the source file

    Returns:
        Detected language name or 'unknown'
    """
    factory = get_global_factory()
    return factory.detect_language(file_path)
