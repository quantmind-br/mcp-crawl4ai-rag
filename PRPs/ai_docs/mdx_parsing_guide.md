# MDX Parsing Implementation Guide

## Critical Context for MDX Parser Implementation

This document provides essential implementation context for adding MDX file support to the file processing pipeline.

## MDX Format Essentials

### Key Differences from Markdown
- **JSX Components**: `<ComponentName prop="value">content</ComponentName>`
- **JSX Expressions**: `{variableName}` or `{expression}`
- **ESM Imports**: `import { Component } from './components'`
- **ESM Exports**: `export const metadata = { title: 'Example' }`
- **Frontmatter**: Optional YAML/JSON metadata block at top

### Critical Parsing Constraints
1. **No Python MDX Libraries**: Must use regex-based approach with existing markdown parser
2. **JSX Self-Closing Tags**: `<Component />` vs paired `<Component></Component>`
3. **Component Names**: Must start with capital letter (A-Z)
4. **Nested Components**: JSX components can contain other JSX components
5. **Prop Types**: Strings (`"value"`), expressions (`{variable}`), booleans (standalone)

## Recommended Implementation Pattern

### Core Regex Patterns

```python
# JSX Component Detection (comprehensive)
jsx_component_pattern = re.compile(
    r'<([A-Z][a-zA-Z0-9]*)'           # Component name (capital first)
    r'(\s+[^>]*?)?'                   # Props (optional)
    r'(?:'
    r'\s*/>'                          # Self-closing
    r'|>'                             # Opening tag
    r'(.*?)'                          # Content (non-greedy)
    r'</\1>'                          # Closing tag
    r')',
    re.DOTALL | re.MULTILINE
)

# Import/Export Detection
import_pattern = re.compile(r'^import\s+.*?from\s+[\'"].*?[\'"]', re.MULTILINE)
export_pattern = re.compile(r'^export\s+.*', re.MULTILINE)

# JSX Expressions
jsx_expression_pattern = re.compile(r'\{([^}]+)\}')
```

### Content Cleaning Strategy

```python
def clean_mdx_content(content):
    """Clean MDX content while preserving semantic meaning"""
    
    # 1. Extract frontmatter using python-frontmatter
    post = frontmatter.loads(content)
    metadata = post.metadata
    body = post.content
    
    # 2. Extract JSX components for metadata
    components = []
    for match in jsx_component_pattern.finditer(body):
        components.append({
            'name': match.group(1),
            'props_raw': match.group(2) or '',
            'content': match.group(3) or '',
            'line_number': body[:match.start()].count('\n') + 1
        })
    
    # 3. Strip JSX but preserve inner text content
    def replace_jsx_component(match):
        component_name = match.group(1)
        inner_content = match.group(3) or ''
        # Keep meaningful text content
        return f"{component_name} {inner_content}".strip()
    
    cleaned = jsx_component_pattern.sub(replace_jsx_component, body)
    
    # 4. Remove imports/exports
    cleaned = import_pattern.sub('', cleaned)
    cleaned = export_pattern.sub('', cleaned)
    
    # 5. Handle JSX expressions - keep variable names
    cleaned = jsx_expression_pattern.sub(r'\1', cleaned)
    
    # 6. Clean up whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned).strip()
    
    return {
        'metadata': metadata,
        'content': cleaned,
        'components': components,
        'preservation_ratio': len(cleaned) / len(content) if content else 0
    }
```

## Required Dependencies

Add to pyproject.toml:
```toml
python-frontmatter = ">=1.1.0"  # For frontmatter parsing
```

## Error Handling Patterns

```python
from ..core.exceptions import ProcessingError

try:
    # MDX processing logic
    result = process_mdx_content(content)
except Exception as e:
    raise ProcessingError(
        f"Error processing MDX file {file_path}: {e}",
        file_path=file_path,
        processor_name="mdx",
        details={'partial_content': content[:500]}
    )
```

## Test Cases to Validate

1. **Basic MDX with Components**:
```mdx
import { Alert } from './components'

# Hello World

<Alert type="info">This is an alert</Alert>
```

2. **Self-Closing Components**:
```mdx
<Image src="/logo.png" alt="Logo" />
```

3. **Nested Components**:
```mdx
<Card>
  <CardHeader>
    <Title>Example</Title>
  </CardHeader>
  <CardBody>
    Content here
  </CardBody>
</Card>
```

4. **JSX Expressions**:
```mdx
export const title = "Dynamic Title"

# {title}

The date is {new Date().toLocaleDateString()}.
```

5. **Frontmatter + JSX**:
```mdx
---
title: "Blog Post"
date: 2024-01-15
---

import { CodeBlock } from './components'

# {title}

<CodeBlock language="javascript">
const x = 1;
</CodeBlock>
```

## Common Pitfalls to Avoid

1. **Greedy Regex Matching**: Use non-greedy `.*?` for component content
2. **Missing DOTALL Flag**: JSX components can span multiple lines
3. **Case Sensitivity**: Only components starting with capital letters are JSX
4. **Malformed JSX**: Handle unclosed tags gracefully
5. **Frontmatter Edge Cases**: Missing closing `---` delimiter

## Performance Considerations

- File size limit: 1MB for MDX files (same as markdown)
- Component extraction limit: Max 50 components per file
- Regex compilation: Compile patterns once in `__init__`
- Content preservation: Aim for >85% of original text content preserved

## Integration with Existing Pipeline

The MDX processor must return ProcessedContent objects that match the expected interface:

```python
processed_content = self.create_processed_content(
    content=cleaned_content,
    content_type="mdx",
    name=filename,
    signature=None,
    line_number=1,
    language="mdx",
)
```

Additional component entries can be added for each JSX component detected.