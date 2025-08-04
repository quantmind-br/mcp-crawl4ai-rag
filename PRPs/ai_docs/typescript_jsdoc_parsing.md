# TypeScript JSDoc/TSDoc Parsing Approaches

This document provides comprehensive guidance for extracting JSDoc/TSDoc comments from TypeScript files for the smart_crawl_github multi-file type enhancement.

## Official Documentation

- **TSDoc Specification**: https://tsdoc.org/
- **JSDoc Documentation**: https://jsdoc.app/
- **TypeScript Handbook**: https://www.typescriptlang.org/docs/handbook/jsdoc-supported-types.html

## Comment Format Standards

### JSDoc Format
```typescript
/**
 * Calculates the area of a rectangle.
 * @param width - The width of the rectangle
 * @param height - The height of the rectangle
 * @returns The area of the rectangle
 * @example
 * ```typescript
 * const area = calculateArea(5, 10);
 * console.log(area); // 50
 * ```
 */
function calculateArea(width: number, height: number): number {
    return width * height;
}
```

### TSDoc Format
```typescript
/**
 * A class representing a database connection.
 * 
 * @public
 */
export class DatabaseConnection {
    /**
     * Connects to the database using the provided configuration.
     * 
     * @param config - The database configuration object
     * @returns A promise that resolves when the connection is established
     * 
     * @throws {@link ConnectionError}
     * When the connection cannot be established
     * 
     * @example
     * ```typescript
     * const db = new DatabaseConnection();
     * await db.connect({ host: 'localhost', port: 5432 });
     * ```
     */
    async connect(config: DatabaseConfig): Promise<void> {
        // Implementation
    }
}
```

## Implementation Approaches

### 1. Regex-Based Approach (Recommended for Simplicity)

```python
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

class TypeScriptDocExtractor:
    """Extract JSDoc/TSDoc comments from TypeScript files using regex."""
    
    # Comprehensive regex pattern for JSDoc comments
    JSDOC_PATTERN = re.compile(
        r'/\*\*\s*\n((?:\s*\*[^\n]*\n)*)\s*\*/',
        re.MULTILINE | re.DOTALL
    )
    
    # Pattern to match function/class/interface declarations
    DECLARATION_PATTERNS = {
        'function': re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?',
            re.MULTILINE
        ),
        'class': re.compile(
            r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?',
            re.MULTILINE
        ),
        'interface': re.compile(
            r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?',
            re.MULTILINE
        ),
        'method': re.compile(
            r'(?:async\s+)?(\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?',
            re.MULTILINE
        )
    }
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract JSDoc comments from TypeScript file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all JSDoc comments
            comments = self._extract_jsdoc_comments(content)
            
            # Associate comments with declarations
            documented_items = self._associate_comments_with_declarations(content, comments)
            
            return {
                'documented_items': documented_items,
                'success': True,
                'metadata': {
                    'total_comments': len(comments),
                    'total_functions': len([item for item in documented_items if item['type'] == 'function']),
                    'total_classes': len([item for item in documented_items if item['type'] == 'class']),
                    'total_interfaces': len([item for item in documented_items if item['type'] == 'interface'])
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'documented_items': []
            }
    
    def _extract_jsdoc_comments(self, content: str) -> List[Dict[str, Any]]:
        """Extract all JSDoc comments from content."""
        comments = []
        
        for match in self.JSDOC_PATTERN.finditer(content):
            comment_text = match.group(1)
            start_pos = match.start()
            
            # Clean up comment text
            lines = comment_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('*'):
                    line = line[1:].strip()
                if line:
                    cleaned_lines.append(line)
            
            cleaned_comment = '\n'.join(cleaned_lines)
            
            # Calculate line number
            line_number = content[:start_pos].count('\n') + 1
            
            comments.append({
                'text': cleaned_comment,
                'start_pos': start_pos,
                'end_pos': match.end(),
                'line_number': line_number
            })
        
        return comments
    
    def _associate_comments_with_declarations(self, content: str, comments: List[Dict]) -> List[Dict[str, Any]]:
        """Associate JSDoc comments with their corresponding declarations."""
        documented_items = []
        
        for comment in comments:
            # Look for declaration after the comment
            after_comment = content[comment['end_pos']:]
            declaration = self._find_next_declaration(after_comment)
            
            if declaration:
                documented_items.append({
                    'type': declaration['type'],
                    'name': declaration['name'],
                    'signature': declaration.get('signature', ''),
                    'docstring': comment['text'],
                    'line_number': comment['line_number'],
                    'language': 'typescript'
                })
        
        return documented_items
    
    def _find_next_declaration(self, content: str) -> Optional[Dict[str, Any]]:
        """Find the next function/class/interface declaration."""
        # Remove leading whitespace and newlines
        content = content.lstrip()
        
        # Try each declaration pattern
        for decl_type, pattern in self.DECLARATION_PATTERNS.items():
            match = pattern.search(content)
            if match and match.start() < 200:  # Must be close to comment
                return {
                    'type': decl_type,
                    'name': match.group(1),
                    'signature': match.group(0)
                }
        
        return None

# Enhanced regex patterns for better matching
class EnhancedTypeScriptDocExtractor(TypeScriptDocExtractor):
    """Enhanced extractor with more sophisticated patterns."""
    
    # More comprehensive function pattern
    FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)',
        re.MULTILINE
    )
    
    # Arrow function pattern
    ARROW_FUNCTION_PATTERN = re.compile(
        r'(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*:\s*[^=]*=>\s*{',
        re.MULTILINE
    )
    
    # Method pattern (inside classes)
    METHOD_PATTERN = re.compile(
        r'(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)(?:\s*:\s*[^{]+)?',
        re.MULTILINE
    )
```

### 2. Tree-sitter Approach (Recommended for Production)

```python
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

class TreeSitterTypeScriptExtractor:
    """Extract JSDoc using Tree-sitter for accurate parsing."""
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter not available. Install with: pip install tree-sitter")
        
        # Note: You need to build the TypeScript grammar
        # See: https://github.com/tree-sitter/tree-sitter-typescript
        try:
            TS_LANGUAGE = Language.build_library(
                'build/languages.so',
                ['tree-sitter-typescript/typescript']
            )
            self.parser = Parser()
            self.parser.set_language(TS_LANGUAGE)
        except Exception as e:
            raise RuntimeError(f"Failed to load TypeScript grammar: {e}")
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract documentation using Tree-sitter parsing."""
        try:
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = self.parser.parse(source_code)
            root_node = tree.root_node
            
            documented_items = []
            self._traverse_tree(root_node, source_code, documented_items)
            
            return {
                'documented_items': documented_items,
                'success': True,
                'metadata': {
                    'parser': 'tree-sitter',
                    'total_items': len(documented_items)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'documented_items': []
            }
    
    def _traverse_tree(self, node, source_code: bytes, documented_items: List[Dict]):
        """Traverse AST and extract documented items."""
        # Look for comment nodes followed by declarations
        if node.type == 'comment' and '/**' in node.text.decode('utf-8'):
            # Process JSDoc comment
            comment_text = self._clean_jsdoc_comment(node.text.decode('utf-8'))
            
            # Find next sibling that's a declaration
            next_sibling = node.next_sibling
            if next_sibling and next_sibling.type in ['function_declaration', 'class_declaration', 'interface_declaration']:
                documented_items.append({
                    'type': next_sibling.type.replace('_declaration', ''),
                    'name': self._extract_name_from_node(next_sibling, source_code),
                    'docstring': comment_text,
                    'line_number': node.start_point[0] + 1,
                    'language': 'typescript'
                })
        
        # Recurse through child nodes
        for child in node.children:
            self._traverse_tree(child, source_code, documented_items)
```

### 3. Hybrid Approach (Recommended Implementation)

```python
class HybridTypeScriptExtractor:
    """Hybrid approach using regex with Tree-sitter fallback."""
    
    def __init__(self):
        self.regex_extractor = TypeScriptDocExtractor()
        self.tree_sitter_available = TREE_SITTER_AVAILABLE
        if self.tree_sitter_available:
            try:
                self.tree_sitter_extractor = TreeSitterTypeScriptExtractor()
            except Exception:
                self.tree_sitter_available = False
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract using best available method."""
        # Try Tree-sitter first for accuracy
        if self.tree_sitter_available:
            try:
                result = self.tree_sitter_extractor.extract_from_file(file_path)
                if result['success']:
                    return result
            except Exception:
                pass  # Fall back to regex
        
        # Fallback to regex approach
        return self.regex_extractor.extract_from_file(file_path)
```

## Error Handling Best Practices

```python
def robust_typescript_extraction(file_path: str) -> Dict[str, Any]:
    """Robust TypeScript documentation extraction."""
    try:
        # Check file size
        file_size = Path(file_path).stat().st_size
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            return {
                'success': False,
                'error': f'File too large: {file_size / 1024 / 1024:.1f}MB',
                'documented_items': []
            }
        
        # Check if file is minified
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if len(first_line) > 1000 and '\n' not in first_line[:1000]:
                return {
                    'success': False,
                    'error': 'File appears to be minified',
                    'documented_items': []
                }
        
        # Extract documentation
        extractor = HybridTypeScriptExtractor()
        return extractor.extract_from_file(file_path)
        
    except UnicodeDecodeError:
        return {
            'success': False,
            'error': 'File encoding not supported',
            'documented_items': []
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'documented_items': []
        }
```

## Metadata Structure for RAG

```python
{
    'content': 'Calculates the area of a rectangle...',  # Clean JSDoc content
    'file_path': 'src/utils.ts',                         # Source file path
    'type': 'function',                                  # function/class/interface/method
    'name': 'calculateArea',                             # Symbol name
    'signature': 'calculateArea(width: number, height: number): number',  # Full signature
    'line_number': 15,                                   # Line number in source
    'language': 'typescript',                            # File type marker
    'tags': ['param', 'returns', 'example']            # JSDoc tags present
}
```

## Performance and Security Notes

- **Regex approach**: Fast but may miss edge cases
- **Tree-sitter approach**: Accurate but requires grammar installation
- **File size limits**: Skip large files (>5MB) to prevent performance issues
- **Minification detection**: Skip minified files as they lack meaningful documentation
- **Encoding handling**: Handle non-UTF-8 files gracefully
- **Memory management**: Use streaming for very large files if needed

## Installation Requirements

For enhanced Tree-sitter support:
```bash
pip install tree-sitter
git clone https://github.com/tree-sitter/tree-sitter-typescript
# Build grammar according to tree-sitter documentation
```

For basic regex approach: No additional dependencies required.