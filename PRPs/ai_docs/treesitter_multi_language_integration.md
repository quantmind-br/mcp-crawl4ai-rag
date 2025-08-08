# Tree-sitter Multi-Language Code Parsing Integration Patterns

## Overview

Tree-sitter provides robust multi-language parsing capabilities that can be integrated into unified data pipelines for code repository processing. This document consolidates best practices and patterns from multiple sources for implementing unified repository processing systems.

**Key Benefits of Tree-sitter for Repository Processing:**
- Concrete Syntax Trees (CSTs) that preserve all source code information
- Incremental parsing with error recovery for malformed code
- Multi-language support with consistent query interface
- Performance optimizations for large codebases

## Core Architecture Patterns

### 1. Parser Factory Pattern (CRITICAL)

```python
from tree_sitter import Language, Parser
from pathlib import Path
from typing import Dict, Optional

class LanguageParserFactory:
    """Manages Tree-sitter parsers for multiple languages with caching."""
    
    def __init__(self, library_path: str = 'build/languages.so'):
        self.library_path = library_path
        self.languages = {}
        self.parsers = {}
        self._init_languages()
    
    def _init_languages(self):
        """Initialize supported languages."""
        try:
            self.languages = {
                'python': Language(self.library_path, 'python'),
                'javascript': Language(self.library_path, 'javascript'), 
                'typescript': Language(self.library_path, 'typescript'),
                'java': Language(self.library_path, 'java')
            }
        except OSError as e:
            raise RuntimeError(f"Failed to load language library: {e}")
    
    def get_parser(self, language: str) -> Optional[Parser]:
        """Get cached parser for language."""
        if language not in self.languages:
            return None
            
        if language not in self.parsers:
            parser = Parser()
            parser.set_language(self.languages[language])
            self.parsers[language] = parser
        
        return self.parsers[language]
    
    def detect_language(self, file_path: str) -> str:
        """Multi-layered language detection."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript', 
            '.java': 'java'
        }
        
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext, 'unknown')
```

### 2. Unified Data Structure Pattern (REQUIRED)

**CRITICAL**: All languages must produce identical output structure to minimize impact on existing Neo4j schema.

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class CodeElement:
    """Standardized code element across all languages."""
    name: str
    full_name: str  # module.name or class.method format
    type: str       # 'class', 'function', 'method', 'import'
    start_line: int
    end_line: int
    language: str
    params: List[Dict[str, Any]] = None  # Parameter details
    params_detailed: List[str] = None    # String representations 
    return_type: str = 'Any'
    attributes: List[Dict[str, str]] = None  # For classes
    methods: List['CodeElement'] = None      # For classes
    imports: List[str] = None                # For modules
    
    def __post_init__(self):
        if self.params is None:
            self.params = []
        if self.params_detailed is None:
            self.params_detailed = []
        if self.attributes is None:
            self.attributes = []
        if self.methods is None:
            self.methods = []
        if self.imports is None:
            self.imports = []

# CRITICAL: This structure must match existing Neo4j schema expectations
def convert_to_neo4j_format(elements: List[CodeElement]) -> Dict[str, Any]:
    """Convert CodeElement list to existing Neo4j schema format."""
    classes = []
    functions = []
    imports = []
    
    for element in elements:
        if element.type == 'class':
            classes.append({
                'name': element.name,
                'full_name': element.full_name,
                'methods': [
                    {
                        'name': method.name,
                        'params': method.params,
                        'params_detailed': method.params_detailed,
                        'return_type': method.return_type,
                        'args': [p['name'] for p in method.params]  # Backwards compatibility
                    }
                    for method in element.methods
                ],
                'attributes': element.attributes
            })
        elif element.type == 'function':
            functions.append({
                'name': element.name,
                'full_name': element.full_name,
                'params': element.params,
                'params_detailed': element.params_detailed,
                'params_list': [f"{p['name']}:{p['type']}" for p in element.params],
                'return_type': element.return_type,
                'args': [p['name'] for p in element.params]  # Backwards compatibility
            })
    
    return {
        'classes': classes,
        'functions': functions,
        'imports': imports
    }
```

### 3. Language-Specific Query Patterns (IMPLEMENTATION CRITICAL)

**Query Structure**: Use standardized capture names for consistency.

```python
# CRITICAL: Query patterns for each supported language
LANGUAGE_QUERIES = {
    'python': {
        'classes': '''
            (class_definition 
                name: (identifier) @class_name
                body: (block) @class_body) @class_def
        ''',
        'functions': '''
            (function_definition 
                name: (identifier) @func_name
                parameters: (parameters) @params
                return_type: (type)? @return_type) @func_def
        ''',
        'methods': '''
            (class_definition
                body: (block
                    (function_definition
                        name: (identifier) @method_name
                        parameters: (parameters) @params
                        return_type: (type)? @return_type) @method_def))
        ''',
        'imports': '''
            [
                (import_statement 
                    name: (dotted_name) @import_name)
                (import_from_statement
                    module_name: (dotted_name) @module_name
                    name: (dotted_name) @import_name)
            ] @import
        '''
    },
    'typescript': {
        'classes': '''
            (class_declaration
                name: (type_identifier) @class_name
                body: (class_body) @class_body) @class_def
        ''',
        'interfaces': '''
            (interface_declaration
                name: (type_identifier) @interface_name
                body: (object_type) @interface_body) @interface_def
        ''',
        'functions': '''
            (function_declaration
                name: (identifier) @func_name
                parameters: (formal_parameters) @params
                return_type: (type_annotation)? @return_type) @func_def
        ''',
        'methods': '''
            [
                (class_declaration
                    body: (class_body
                        (method_definition
                            name: (property_identifier) @method_name
                            parameters: (formal_parameters) @params
                            return_type: (type_annotation)? @return_type) @method_def))
                (interface_declaration
                    body: (object_type
                        (method_signature
                            name: (property_identifier) @method_name
                            parameters: (formal_parameters) @params
                            return_type: (type_annotation)? @return_type) @method_sig))
            ]
        ''',
        'imports': '''
            [
                (import_statement
                    source: (string) @module_name)
                (import_statement
                    import_clause: (import_clause
                        name: (identifier) @import_name)
                    source: (string) @module_name)
            ] @import
        '''
    },
    'java': {
        'classes': '''
            (class_declaration
                name: (identifier) @class_name
                body: (class_body) @class_body) @class_def
        ''',
        'functions': '''
            (method_declaration
                name: (identifier) @method_name
                parameters: (formal_parameters) @params
                type: (type_identifier)? @return_type) @method_def
        ''',
        'imports': '''
            (import_declaration
                (scoped_identifier) @import_name) @import
        '''
    }
}
```

### 4. Tree-sitter Language Parser Implementation

```python
class TreeSitterLanguageParser:
    """Base parser using Tree-sitter for multi-language support."""
    
    def __init__(self, factory: LanguageParserFactory):
        self.factory = factory
        self.query_cache = {}  # Performance optimization
    
    def parse(self, file_content: str, file_path: str, language: str) -> Dict[str, Any]:
        """Parse file and return standardized structure."""
        parser = self.factory.get_parser(language)
        if not parser:
            return None
        
        try:
            # Parse content
            content_bytes = file_content.encode('utf-8')
            tree = parser.parse(content_bytes)
            
            if tree.root_node.has_error:
                # Log warning but continue with partial results
                print(f"Warning: Parse errors in {file_path}")
            
            # Extract elements using language-specific queries
            elements = self._extract_elements(
                tree, language, content_bytes, file_path
            )
            
            # Convert to Neo4j-compatible format
            return convert_to_neo4j_format(elements)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_elements(self, tree, language: str, content: bytes, file_path: str) -> List[CodeElement]:
        """Extract code elements using Tree-sitter queries."""
        elements = []
        lang_obj = self.factory.languages[language]
        module_name = self._get_module_name(file_path)
        
        # Get cached queries for performance
        queries = self._get_cached_queries(lang_obj, language)
        
        # Extract classes
        for node, _ in queries['classes'].captures(tree.root_node):
            class_element = self._extract_class(node, content, module_name, language)
            if class_element:
                elements.append(class_element)
        
        # Extract top-level functions  
        for node, _ in queries['functions'].captures(tree.root_node):
            func_element = self._extract_function(node, content, module_name, language)
            if func_element:
                elements.append(func_element)
        
        return elements
    
    def _get_cached_queries(self, lang_obj, language: str) -> Dict:
        """Get cached compiled queries for performance."""
        cache_key = f"{language}_queries"
        if cache_key not in self.query_cache:
            patterns = LANGUAGE_QUERIES.get(language, {})
            compiled_queries = {}
            
            for query_type, pattern in patterns.items():
                try:
                    compiled_queries[query_type] = lang_obj.query(pattern)
                except Exception as e:
                    print(f"Query compilation error for {language}.{query_type}: {e}")
                    # Create empty query as fallback
                    compiled_queries[query_type] = lang_obj.query("(ERROR) @error")
            
            self.query_cache[cache_key] = compiled_queries
        
        return self.query_cache[cache_key]
```

## Critical Implementation Notes

### 1. Error Handling Strategy

```python
def safe_parse_with_fallback(self, content: str, file_path: str) -> Dict[str, Any]:
    """Parse with multiple fallback strategies."""
    language = self.factory.detect_language(file_path)
    
    # Primary: Tree-sitter parsing
    try:
        result = self.parse(content, file_path, language)
        if result and (result['classes'] or result['functions']):
            return result
    except Exception as e:
        print(f"Tree-sitter parsing failed for {file_path}: {e}")
    
    # Fallback: AST parsing for Python files
    if language == 'python':
        try:
            return self._fallback_ast_parse(content, file_path)
        except Exception as e:
            print(f"AST fallback failed for {file_path}: {e}")
    
    # Final fallback: Return empty structure
    return {'classes': [], 'functions': [], 'imports': []}
```

### 2. Performance Optimization Patterns

```python
class PerformanceOptimizedParser:
    def __init__(self):
        self.query_cache = {}        # Cache compiled queries
        self.parser_pool = {}        # Reuse parser instances
        self.size_limits = {         # File size limits by language
            'python': 2_000_000,     # 2MB
            'typescript': 1_000_000, # 1MB  
            'java': 3_000_000        # 3MB
        }
    
    def should_parse_file(self, file_path: str, file_size: int) -> bool:
        """Check if file should be parsed based on size limits."""
        language = self.factory.detect_language(file_path)
        limit = self.size_limits.get(language, 1_000_000)
        return file_size <= limit
```

### 3. Grammar Build Script (REQUIRED)

```python
#!/usr/bin/env python3
"""Build script for Tree-sitter language grammars."""

import os
import subprocess
import shutil
from pathlib import Path
from tree_sitter import Language

def build_grammars():
    """Build all required language parsers into shared library."""
    
    # Grammar repositories (clone if not exists)
    grammars = {
        'python': 'https://github.com/tree-sitter/tree-sitter-python',
        'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript', 
        'typescript': 'https://github.com/tree-sitter/tree-sitter-typescript',
        'java': 'https://github.com/tree-sitter/tree-sitter-java'
    }
    
    # Create grammars directory
    grammars_dir = Path('knowledge_graphs/grammars')
    grammars_dir.mkdir(exist_ok=True)
    
    # Clone grammars if not exists
    repo_paths = []
    for name, url in grammars.items():
        repo_path = grammars_dir / f"tree-sitter-{name}"
        if not repo_path.exists():
            print(f"Cloning {name} grammar...")
            subprocess.run(['git', 'clone', url, str(repo_path)], check=True)
        
        if name == 'typescript':
            # TypeScript has subdirectories
            repo_paths.extend([
                str(repo_path / 'typescript'),
                str(repo_path / 'tsx')
            ])
        else:
            repo_paths.append(str(repo_path))
    
    # Build shared library
    build_dir = Path('build')
    build_dir.mkdir(exist_ok=True)
    
    print("Building shared library...")
    Language.build_library(
        str(build_dir / 'languages.so'),  # Output file
        repo_paths                         # Grammar paths
    )
    
    print("✅ Grammar build complete!")

if __name__ == '__main__':
    build_grammars()
```

## Installation Requirements

### 1. System Dependencies
- **Linux/macOS**: GCC or Clang compiler
- **Windows**: Microsoft Visual C++ Build Tools or Visual Studio

### 2. Python Dependencies
```toml
# Add to pyproject.toml
dependencies = [
    # ... existing dependencies
    "tree-sitter>=0.21.0",
]
```

### 3. Build Process
```bash
# 1. Install tree-sitter
pip install tree-sitter

# 2. Build grammars (run once during setup)  
python knowledge_graphs/build_grammars.py

# 3. Verify build
python -c "from tree_sitter import Language; print(Language('build/languages.so', 'python'))"
```

## Integration with Existing Codebase

### 1. Replace AST Logic in `parse_repo_into_neo4j.py`

```python
# OLD (AST-based)
def analyze_python_file(self, file_path: Path, repo_root: Path, project_modules: Set[str]):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    tree = ast.parse(content)  # <- REPLACE THIS
    # ... rest of AST logic

# NEW (Tree-sitter based)
def analyze_file(self, file_path: Path, repo_root: Path, project_modules: Set[str]):
    language = self.parser_factory.detect_language(str(file_path))
    if language == 'unknown':
        return None
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Use Tree-sitter parser
    result = self.tree_sitter_parser.parse(content, str(file_path), language)
    if not result:
        return None
    
    # Convert to existing format for Neo4j compatibility
    relative_path = str(file_path.relative_to(repo_root))
    module_name = self._get_importable_module_name(file_path, repo_root, relative_path)
    
    return {
        'module_name': module_name,
        'file_path': relative_path,
        'classes': result['classes'],
        'functions': result['functions'], 
        'imports': result['imports'],
        'line_count': len(content.splitlines())
    }
```

### 2. Update File Discovery in `github_processor.py`

```python
# Extend SUPPORTED_EXTENSIONS in MultiFileDiscovery
SUPPORTED_EXTENSIONS = {
    ".md", ".markdown", ".mdown", ".mkd",  # Markdown
    ".py",                                  # Python  
    ".ts", ".tsx",                         # TypeScript
    ".js", ".jsx",                         # JavaScript  
    ".java",                               # Java
    ".json", ".yaml", ".yml", ".toml",     # Configuration
}
```

## Testing Strategy

### 1. Unit Tests for Each Language

```python
import pytest
from knowledge_graphs.tree_sitter_parser import TreeSitterParser

class TestTreeSitterParsing:
    
    def test_python_class_extraction(self):
        code = '''
class TestClass:
    def method_one(self, param: str) -> bool:
        return True
        
    def method_two(self):
        pass
'''
        parser = TreeSitterParser()
        result = parser.parse(code, 'test.py', 'python')
        
        assert len(result['classes']) == 1
        assert result['classes'][0]['name'] == 'TestClass'
        assert len(result['classes'][0]['methods']) == 2
    
    def test_typescript_interface_extraction(self):
        code = '''
interface User {
    name: string;
    getId(): number;
}
'''
        parser = TreeSitterParser()
        result = parser.parse(code, 'test.ts', 'typescript')
        
        # Should be extracted as class-like structure
        assert len(result['classes']) == 1
        assert result['classes'][0]['name'] == 'User'
```

### 2. Integration Test with Neo4j

```python
def test_multi_language_neo4j_integration():
    """Test that all languages populate Neo4j correctly."""
    # Create test repository with mixed languages
    test_files = {
        'test.py': 'class PythonClass: pass',
        'test.ts': 'class TypeScriptClass {}',
        'test.java': 'public class JavaClass {}'
    }
    
    # Run parsing
    extractor = DirectNeo4jExtractor()
    # ... test Neo4j population
    
    # Verify all languages are represented
    # Query Neo4j for classes from each language
```

This integration guide provides the critical patterns and implementation details needed to successfully integrate Tree-sitter into the existing AST-based system while maintaining compatibility with the current Neo4j schema.