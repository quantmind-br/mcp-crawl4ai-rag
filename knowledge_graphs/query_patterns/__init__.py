"""
Query Patterns Module

Centralized access to Tree-sitter query patterns for all supported programming languages.
This module provides a unified interface for accessing language-specific S-expression
queries used in the Tree-sitter multi-language code analysis integration.
"""

from typing import Dict, List, Optional

# Import all language-specific query modules
from knowledge_graphs.query_patterns import python_queries
from knowledge_graphs.query_patterns import javascript_queries
from knowledge_graphs.query_patterns import typescript_queries
from knowledge_graphs.query_patterns import java_queries
from knowledge_graphs.query_patterns import go_queries
from knowledge_graphs.query_patterns import c_queries

# Registry of all supported languages and their query modules
LANGUAGE_MODULES = {
    "python": python_queries,
    "javascript": javascript_queries,
    "typescript": typescript_queries,
    "java": java_queries,
    "go": go_queries,
    "rust": go_queries,  # Rust can share some Go patterns initially
    "c": c_queries,
    "cpp": c_queries,  # C++ can share C patterns initially
    "c_sharp": java_queries,  # C# can share Java patterns initially
    "php": python_queries,  # PHP can share Python patterns initially
    "ruby": python_queries,  # Ruby can share Python patterns initially
    "kotlin": java_queries,  # Kotlin can share Java patterns initially
}


def get_queries_for_language(language: str) -> Dict[str, str]:
    """
    Get all query patterns for a specific language.

    Args:
        language: The programming language name (e.g., 'python', 'typescript')

    Returns:
        Dictionary of query_type -> S-expression query string

    Raises:
        ValueError: If language is not supported
    """
    if language not in LANGUAGE_MODULES:
        supported = ", ".join(LANGUAGE_MODULES.keys())
        raise ValueError(f"Language '{language}' not supported. Available: {supported}")

    module = LANGUAGE_MODULES[language]
    return module.get_all_queries()


def get_query(language: str, query_type: str) -> str:
    """
    Get a specific query pattern for a language.

    Args:
        language: The programming language name
        query_type: The type of query to retrieve (e.g., 'classes', 'functions')

    Returns:
        The S-expression query string

    Raises:
        ValueError: If language is not supported
        KeyError: If query_type is not available for the language
    """
    if language not in LANGUAGE_MODULES:
        supported = ", ".join(LANGUAGE_MODULES.keys())
        raise ValueError(f"Language '{language}' not supported. Available: {supported}")

    module = LANGUAGE_MODULES[language]
    return module.get_query(query_type)


def get_supported_languages() -> List[str]:
    """
    Get list of all supported programming languages.

    Returns:
        List of language names that have query patterns available
    """
    return list(LANGUAGE_MODULES.keys())


def get_common_query_types() -> List[str]:
    """
    Get list of query types that are commonly available across languages.

    Returns:
        List of common query type names
    """
    return ["classes", "functions", "methods", "imports"]


def language_supports_query(language: str, query_type: str) -> bool:
    """
    Check if a language supports a specific query type.

    Args:
        language: The programming language name
        query_type: The type of query to check

    Returns:
        True if the language supports the query type, False otherwise
    """
    try:
        queries = get_queries_for_language(language)
        return query_type in queries
    except ValueError:
        return False


def get_language_constructs(language: str) -> Optional[Dict]:
    """
    Get language-specific constructs and conventions.

    Args:
        language: The programming language name

    Returns:
        Dictionary of language constructs or None if not available
    """
    if language not in LANGUAGE_MODULES:
        return None

    module = LANGUAGE_MODULES[language]
    constructs_attr = f"{language.upper()}_CONSTRUCTS"

    # Try different naming conventions
    for attr_name in [constructs_attr, f"{language}_CONSTRUCTS", "CONSTRUCTS"]:
        if hasattr(module, attr_name):
            return getattr(module, attr_name)

    return None


def validate_identifier_for_language(language: str, identifier: str) -> bool:
    """
    Validate an identifier according to language-specific rules.

    Args:
        language: The programming language name
        identifier: The identifier to validate

    Returns:
        True if identifier is valid for the language, False otherwise
    """
    if language not in LANGUAGE_MODULES:
        return False

    module = LANGUAGE_MODULES[language]

    # Try to find a validation function
    validate_func_name = f"validate_{language}_identifier"
    if hasattr(module, validate_func_name):
        validate_func = getattr(module, validate_func_name)
        return validate_func(identifier)

    # Fallback to generic validation
    import re

    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier))


# Export commonly used functions and constants
__all__ = [
    "get_queries_for_language",
    "get_query",
    "get_supported_languages",
    "get_common_query_types",
    "language_supports_query",
    "get_language_constructs",
    "validate_identifier_for_language",
    "LANGUAGE_MODULES",
]
