# Python AST Docstring Extraction Best Practices

This document provides comprehensive guidance for extracting docstrings from Python files using the AST module, specifically for the smart_crawl_github multi-file type enhancement.

## Official Documentation

- **Python 3.13 AST Documentation**: https://docs.python.org/3/library/ast.html
- **Source Code**: https://github.com/python/cpython/tree/3.13/Lib/ast.py
- **Key Function**: `ast.get_docstring(node, clean=True)` - Official built-in function for docstring extraction

## Core Implementation Pattern

```python
import ast
from typing import List, Dict, Any, Optional

class PythonDocstringExtractor:
    """Extract docstrings from Python files using AST parsing."""
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract all docstrings from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=file_path)
            docstrings = []
            
            # Extract module docstring
            module_doc = ast.get_docstring(tree, clean=True)
            if module_doc:
                docstrings.append({
                    'type': 'module',
                    'name': file_path,
                    'docstring': module_doc,
                    'lineno': 1,
                    'signature': None,
                    'parent': None
                })
            
            # Extract class and function docstrings
            for node in ast.walk(tree):
                self._extract_node_docstring(node, docstrings)
            
            return {
                'docstrings': docstrings,
                'success': True,
                'metadata': {
                    'total_functions': len([d for d in docstrings if d['type'] == 'function']),
                    'total_classes': len([d for d in docstrings if d['type'] == 'class']),
                    'total_methods': len([d for d in docstrings if d['type'] == 'method'])
                }
            }
            
        except SyntaxError as e:
            return {
                'success': False,
                'error': f'Syntax error at line {e.lineno}: {e.msg}',
                'docstrings': []
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'docstrings': []
            }
    
    def _extract_node_docstring(self, node: ast.AST, docstrings: List[Dict]) -> None:
        """Extract docstring from AST node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                docstrings.append({
                    'type': 'function',  # or 'method' if inside class
                    'name': node.name,
                    'docstring': docstring,
                    'lineno': node.lineno,
                    'signature': self._extract_signature(node),
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                })
        
        elif isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node, clean=True)
            if docstring:
                docstrings.append({
                    'type': 'class',
                    'name': node.name,
                    'docstring': docstring,
                    'lineno': node.lineno,
                    'signature': None,
                    'bases': [self._ast_to_string(base) for base in node.bases]
                })
    
    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature with type annotations."""
        try:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            
            signature = f"({', '.join(args)})"
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"
            
            return signature
        except Exception:
            return "(signature_extraction_failed)"
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string."""
        try:
            return ast.unparse(node)
        except Exception:
            return str(type(node).__name__)
```

## Error Handling Requirements

1. **MANDATORY**: Use `ast.get_docstring()` - never regex for Python docstrings
2. **Handle SyntaxError**: Files with syntax errors should be skipped gracefully
3. **Handle encoding issues**: Use UTF-8 with error handling
4. **Preserve file processing**: Skip problematic files but continue processing others

## Metadata Structure for RAG

```python
{
    'content': docstring_text,           # Clean docstring content
    'file_path': 'src/module.py',        # Source file path
    'type': 'function',                  # module/class/function/method
    'name': 'function_name',             # Symbol name
    'signature': 'func(arg: int) -> str', # Function signature with types
    'lineno': 45,                        # Line number in source
    'parent_class': 'ClassName',         # For methods only
    'is_async': False,                   # For async functions
    'language': 'python'                 # File type marker
}
```

## Integration with Chunking Pipeline

- **Individual chunks**: Each docstring becomes a separate chunk
- **Rich metadata**: Include file path, symbol name, signature for context
- **Content focus**: Index the docstring content, not the code
- **Context preservation**: Metadata provides full context for RAG queries

## Performance Considerations

- **AST parsing**: Fast and memory-efficient for most Python files
- **Error isolation**: Skip problematic files without affecting others
- **Lazy evaluation**: Only parse files that pass basic validation checks
- **Size limits**: Skip extremely large Python files (>1MB recommended)

## Security Notes

- **AST parsing is safe**: No code execution, unlike `eval()` or `exec()`
- **File size limits**: Prevent DoS attacks with large files
- **Encoding safety**: Handle encoding errors gracefully
- **Path validation**: Ensure file paths are within expected directories