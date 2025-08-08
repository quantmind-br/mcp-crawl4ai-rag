"""
Knowledge Graph Package

This package contains all knowledge graph related functionality including:
- Multi-language code parsing with Tree-sitter
- Neo4j graph population and querying
- AI hallucination detection and validation
- Parser factory and language detection
"""

# Make key components available at package level
try:
    from knowledge_graphs.parser_factory import ParserFactory, get_global_factory
    from knowledge_graphs.language_parser import LanguageParser, ParseResult
    from knowledge_graphs.ai_script_analyzer import AIScriptAnalyzer
    from knowledge_graphs.parse_repo_into_neo4j import Neo4jCodeAnalyzer

    __all__ = [
        "ParserFactory",
        "get_global_factory",
        "LanguageParser",
        "ParseResult",
        "AIScriptAnalyzer",
        "Neo4jCodeAnalyzer",
    ]
except ImportError:
    # Allow package to be imported even if dependencies are missing
    __all__ = []
