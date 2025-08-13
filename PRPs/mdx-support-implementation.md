# MDX File Support Implementation

## Goal

**Feature Goal**: Add native MDX file processing support to the GitHub crawling and indexing tools (`smart_crawl_github` and `index_github_repository`) with identical functionality to existing Markdown file processing.

**Deliverable**: Complete MDXProcessor class that extends BaseFileProcessor, integrates with ProcessorFactory, and processes .mdx files for both RAG (Qdrant) and Knowledge Graph (Neo4j) storage systems.

**Success Definition**: MDX files are discovered, processed, and indexed with >95% success rate, preserving >85% of text content while extracting JSX component metadata, with zero breaking changes to existing .md file processing.

## User Persona

**Target User**: AI agents and coding assistants using MCP tools for repository analysis

**Use Case**: Indexing modern React-based documentation repositories (Docusaurus, Nextra, Storybook) that use MDX files containing both Markdown content and interactive JSX components

**User Journey**: 
1. Agent calls `smart_crawl_github` or `index_github_repository` with file_types including ".mdx"
2. Tool discovers MDX files in repository alongside other file types
3. MDX files are processed to extract content and JSX component metadata
4. Content is indexed in both vector database (RAG) and knowledge graph (KG) systems
5. Agent can search and retrieve MDX-sourced content through existing RAG tools

**Pain Points Addressed**: Currently MDX files are ignored during repository indexing, causing incomplete content coverage for modern documentation sites and React component libraries

## Why

- **Modern Documentation Adoption**: MDX is increasingly used in React-based documentation (Docusaurus, Nextra, Storybook docs)
- **Content Completeness**: GitHub repositories with MDX files lose valuable documentation during indexing
- **Component Library Support**: Design systems and component libraries heavily use MDX for documentation
- **Zero Breaking Changes**: Implementation extends existing patterns without affecting current .md processing

## What

Add MDX file processing capabilities that:
- Recognize .mdx file extensions during repository scanning
- Extract Markdown content while preserving semantic meaning
- Detect and catalog JSX components with metadata
- Handle imports/exports and frontmatter appropriately
- Integrate seamlessly with existing Qdrant and Neo4j storage
- Maintain identical performance characteristics to Markdown processing

### Success Criteria

- [ ] .mdx files are discovered and processed during repository indexing
- [ ] JSX components are stripped from content while preserving inner text
- [ ] Component metadata is extracted and stored for enhanced search
- [ ] Processing time ≤ 2x standard markdown processing time
- [ ] Content preservation rate >85% of original meaningful text
- [ ] Zero impact on existing .md file processing functionality
- [ ] All validation gates pass (linting, type checking, tests)

## All Needed Context

### Context Completeness Check

_This PRP contains complete implementation context including exact file paths, method signatures, architectural patterns, external research, and validation approaches. An implementer with no prior codebase knowledge can successfully implement this feature._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- docfile: PRPs/ai_docs/mdx_parsing_guide.md
  why: Complete MDX parsing implementation patterns, regex approaches, and error handling
  section: "Recommended Implementation Pattern"
  critical: Contains tested regex patterns and content cleaning strategies

- url: https://mdxjs.com/docs/what-is-mdx/#mdx-syntax
  why: Official MDX syntax specification for component recognition patterns
  critical: Defines JSX component naming rules (must start with capital letter)

- file: src/features/github/processors/markdown_processor.py
  why: Exact pattern to follow for processor implementation - same structure, naming, error handling
  pattern: BaseFileProcessor extension, _process_file_impl method, create_processed_content usage
  gotcha: Must use same file size limits and validation patterns

- file: src/features/github/processors/base_processor.py
  why: Abstract base class definition with required method signatures
  pattern: Abstract method _process_file_impl, helper methods available, error handling approach
  gotcha: Must call super().__init__ with correct parameters

- file: src/features/github/processors/processor_factory.py
  why: Registration pattern for new processors in factory system
  pattern: _create_default_registry method, registry.register calls
  gotcha: Import must be added and processor registered with exact name "mdx"

- file: src/features/github/config/settings.py
  why: Configuration updates needed for language mappings and processor priority
  pattern: LANGUAGE_MAPPINGS dict, PROCESSOR_PRIORITY dict additions
  gotcha: Must add ".mdx": "mdx" mapping and priority level 10 (same as markdown)

- file: src/features/github/core/models.py
  why: ProcessedContent dataclass structure for return values
  pattern: Required fields, dataclass decorator, type hints
  gotcha: content_type should be "mdx", language should be "mdx"

- file: tests/unit/tools/test_github_tools.py
  why: Test pattern for verifying processor integration with GitHub tools
  pattern: Mock setup, assertion patterns, error case testing
  gotcha: Must test both smart_crawl_github and index_github_repository tools
```

### Current Codebase Tree

```bash
src/
├── features/github/
│   ├── processors/
│   │   ├── base_processor.py          # Abstract base class to extend
│   │   ├── markdown_processor.py      # Exact pattern to follow
│   │   ├── processor_factory.py       # Registration point
│   │   └── mdx_processor.py          # NEW FILE TO CREATE
│   ├── config/
│   │   └── settings.py               # Configuration updates needed
│   └── core/
│       ├── models.py                 # ProcessedContent dataclass
│       └── exceptions.py             # Error types to use
├── tools/
│   └── github_tools.py               # Integration points (processor_map)
└── services/
    └── unified_indexing_service.py   # Will automatically support MDX
tests/
├── unit/
│   ├── tools/
│   │   └── test_github_tools.py      # Add MDX test cases
│   └── features/github/processors/
│       └── test_mdx_processor.py     # NEW TEST FILE TO CREATE
```

### Desired Codebase Tree with New Files

```bash
src/features/github/processors/
└── mdx_processor.py                  # MDXProcessor class extending BaseFileProcessor

tests/unit/features/github/processors/
└── test_mdx_processor.py            # Unit tests for MDXProcessor class

# MODIFIED FILES:
src/features/github/processors/processor_factory.py  # Add MDX registration
src/features/github/config/settings.py              # Add .mdx mappings
src/tools/github_tools.py                          # Add .mdx to processor_map
```

### Known Gotchas of Codebase & Library Quirks

```python
# CRITICAL: BaseFileProcessor requires specific constructor pattern
super().__init__(
    name="mdx",  # Must match registration name exactly
    supported_extensions=[".mdx"],  # List with dot notation
    max_file_size=1_000_000,  # 1MB limit (same as markdown)
)

# CRITICAL: ProcessedContent creation pattern must match exactly
processed_content = self.create_processed_content(
    content=cleaned_content,    # Cleaned text content (required)
    content_type="mdx",         # Type identifier (required)
    name=filename,              # File name (required)
    signature=None,             # Method signature (optional, use None)
    line_number=1,              # Starting line (required)
    language="mdx",             # Language identifier (required)
)

# CRITICAL: Error handling must use project exception types
from ..core.exceptions import ProcessingError
raise ProcessingError(
    f"Error processing MDX file {file_path}: {e}",
    file_path=file_path,
    processor_name=self.name,
)

# CRITICAL: File reading must use base class helper
content = self.read_file_content(file_path)  # Handles encoding and errors

# CRITICAL: Content validation follows base class pattern
if not self.validate_content(content, min_length=50):
    return []  # Return empty list, not None

# GOTCHA: Regex patterns must compile once in __init__ for performance
self.jsx_pattern = re.compile(r'<([A-Z][a-zA-Z0-9]*)', re.DOTALL)

# GOTCHA: Windows console encoding - avoid Unicode in debug prints
# Use ASCII-only strings in any print/log statements
```

## Implementation Blueprint

### Data Models and Structure

MDX processing leverages existing ProcessedContent model with enhanced metadata:

```python
# Core data structure (already exists)
@dataclass
class ProcessedContent:
    content: str                    # Cleaned text content
    content_type: str               # "mdx" for main content, "jsx_component" for components
    name: str                       # Filename or component name
    signature: Optional[str] = None # JSX component signature
    line_number: int = 1           # Line number in source
    language: str = "text"         # "mdx" or "jsx"

# Enhanced metadata through component extraction
mdx_metadata = {
    'jsx_components_count': int,
    'jsx_components': List[Dict],
    'imports_detected': List[str],
    'exports_detected': List[str],
    'content_preservation_ratio': float
}
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/features/github/processors/mdx_processor.py
  - IMPLEMENT: MDXProcessor class extending BaseFileProcessor
  - FOLLOW pattern: src/features/github/processors/markdown_processor.py (class structure, method signatures)
  - NAMING: MDXProcessor class, _process_file_impl method (exact name required)
  - DEPENDENCIES: Use python-frontmatter for frontmatter parsing, regex for JSX extraction
  - PLACEMENT: Same directory as other processors

Task 2: MODIFY src/features/github/processors/processor_factory.py
  - IMPLEMENT: Import MDXProcessor and register in _create_default_registry
  - FOLLOW pattern: existing processor imports and registry.register calls
  - NAMING: registry.register("mdx", MDXProcessor) - exact name "mdx" required
  - DEPENDENCIES: Import from Task 1
  - PLACEMENT: Add to existing imports and registry

Task 3: MODIFY src/features/github/config/settings.py
  - IMPLEMENT: Add .mdx to LANGUAGE_MAPPINGS and PROCESSOR_PRIORITY
  - FIND pattern: existing mappings in LANGUAGE_MAPPINGS dict
  - ADD: ".mdx": "mdx" to LANGUAGE_MAPPINGS, "mdx": 10 to PROCESSOR_PRIORITY
  - PRESERVE: All existing mappings and settings

Task 4: MODIFY src/tools/github_tools.py  
  - INTEGRATE: Add .mdx to processor_map dictionary
  - FIND pattern: existing processor_map entries around line 158
  - ADD: ".mdx": MarkdownProcessor, # Use existing MarkdownProcessor for backward compatibility
  - PRESERVE: All existing processor mappings

Task 5: CREATE tests/unit/features/github/processors/test_mdx_processor.py
  - IMPLEMENT: Unit tests for MDXProcessor class methods
  - FOLLOW pattern: tests/unit/tools/test_github_tools.py (test structure, fixtures, assertions)
  - NAMING: test_mdx_processor_* function naming convention
  - COVERAGE: Happy path, error cases, JSX component extraction
  - PLACEMENT: New test file in processors test directory

Task 6: MODIFY tests/unit/tools/test_github_tools.py
  - INTEGRATE: Add MDX file testing to existing GitHub tools tests
  - FIND pattern: existing file type testing scenarios
  - ADD: Test cases with .mdx files in file_types_to_index parameter
  - PRESERVE: All existing test cases and functionality
```

### Implementation Patterns & Key Details

```python
# MDXProcessor implementation pattern (complete structure)
import os
import re
from typing import List

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent
from ..core.exceptions import ProcessingError

class MDXProcessor(BaseFileProcessor):
    """Process MDX files with JSX component extraction."""

    def __init__(self):
        """Initialize MDX processor with regex patterns."""
        super().__init__(
            name="mdx",
            supported_extensions=[".mdx"],
            max_file_size=1_000_000,  # 1MB limit (same as markdown)
        )
        
        # PATTERN: Compile regex patterns once for performance
        self.jsx_component_pattern = re.compile(
            r'<([A-Z][a-zA-Z0-9]*)\s*([^>]*?)(?:/>|>(.*?)</\1>)',
            re.DOTALL | re.MULTILINE
        )
        
        self.import_pattern = re.compile(r'^import\s+.*?from\s+[\'"].*?[\'"]', re.MULTILINE)
        self.export_pattern = re.compile(r'^export\s+.*', re.MULTILINE)

    def _process_file_impl(self, file_path: str, relative_path: str, **kwargs) -> List[ProcessedContent]:
        """Process MDX file and extract content and components."""
        try:
            # PATTERN: Use base class file reading with error handling
            content = self.read_file_content(file_path)

            # PATTERN: Validate content with same criteria as markdown
            if not self.validate_content(content, min_length=50):
                return []

            filename = os.path.basename(file_path)
            extracted_items = []

            # CRITICAL: Clean content while preserving semantic meaning
            cleaned_content = self._clean_mdx_content(content)
            
            # PATTERN: Create main content entry using base class helper
            processed_content = self.create_processed_content(
                content=cleaned_content,
                content_type="mdx",
                name=filename,
                signature=None,
                line_number=1,
                language="mdx",
            )
            extracted_items.append(processed_content)

            # GOTCHA: Extract JSX components for enhanced metadata
            jsx_components = self._extract_jsx_components(content)
            extracted_items.extend(jsx_components)

            return extracted_items

        except Exception as e:
            # PATTERN: Use project exception types with file context
            raise ProcessingError(
                f"Error processing MDX file {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    def _clean_mdx_content(self, content: str) -> str:
        """Clean MDX content for text indexing."""
        # CRITICAL: Use patterns from PRPs/ai_docs/mdx_parsing_guide.md
        # Strip JSX components but preserve inner text content
        # Remove imports/exports
        # Handle frontmatter appropriately
        # Return cleaned text content (see guide for complete implementation)

    def _extract_jsx_components(self, content: str) -> List[ProcessedContent]:
        """Extract JSX components as separate entities."""
        # CRITICAL: Follow pattern from mdx_parsing_guide.md
        # Return list of ProcessedContent objects for each component
        # Use content_type="jsx_component" and language="jsx"
```

### Integration Points

```yaml
PROCESSOR_REGISTRATION:
  - file: src/features/github/processors/processor_factory.py
  - method: _create_default_registry
  - pattern: "registry.register('mdx', MDXProcessor)"

CONFIGURATION:
  - file: src/features/github/config/settings.py  
  - add to: LANGUAGE_MAPPINGS dict
  - pattern: '".mdx": "mdx"'
  - add to: PROCESSOR_PRIORITY dict
  - pattern: '"mdx": 10'

TOOL_INTEGRATION:
  - file: src/tools/github_tools.py
  - add to: processor_map dict (line ~158)
  - pattern: '".mdx": MarkdownProcessor,'

DEPENDENCIES:
  - add to: pyproject.toml dependencies
  - pattern: 'python-frontmatter>=1.1.0'
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
uv run ruff check src/features/github/processors/mdx_processor.py --fix
uv run mypy src/features/github/processors/mdx_processor.py
uv run ruff format src/features/github/processors/mdx_processor.py

# Validate modified files
uv run ruff check src/features/github/processors/processor_factory.py --fix
uv run ruff check src/features/github/config/settings.py --fix
uv run ruff check src/tools/github_tools.py --fix

# Project-wide validation
uv run ruff check src/ --fix
uv run mypy src/
uv run ruff format src/

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test new MDX processor specifically
uv run pytest tests/unit/features/github/processors/test_mdx_processor.py -v

# Test integration with GitHub tools
uv run pytest tests/unit/tools/test_github_tools.py::test_smart_crawl_github_mdx -v
uv run pytest tests/unit/tools/test_github_tools.py::test_index_github_repository_mdx -v

# Full processor test suite
uv run pytest tests/unit/features/github/processors/ -v

# Coverage validation for new code
uv run pytest tests/unit/features/github/processors/test_mdx_processor.py --cov=src.features.github.processors.mdx_processor --cov-report=term-missing

# Expected: All tests pass with >90% coverage for new code
```

### Level 3: Integration Testing (System Validation)

```bash
# Start MCP server for integration testing
uv run -m src &
SERVER_PID=$!
sleep 3  # Allow startup time

# Test MDX processing through MCP tools
echo '{"method": "tools/call", "params": {"name": "smart_crawl_github", "arguments": {"repo_url": "https://github.com/facebook/docusaurus", "file_types_to_index": [".mdx"], "max_files": 5}}}' | \
  uv run python -m src

# Test unified indexing with MDX files
echo '{"method": "tools/call", "params": {"name": "index_github_repository", "arguments": {"repo_url": "https://github.com/vercel/nextra", "file_types": [".mdx"], "destination": "both", "max_files": 10}}}' | \
  uv run python -m src

# Cleanup
kill $SERVER_PID

# Expected: Successful MDX file processing with component extraction
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Test with real-world MDX repositories
MDX_TEST_REPOS=(
  "https://github.com/facebook/docusaurus"
  "https://github.com/vercel/nextra"  
  "https://github.com/storybookjs/storybook"
)

for repo in "${MDX_TEST_REPOS[@]}"; do
  echo "Testing MDX processing with $repo"
  uv run python -c "
import asyncio
from src.tools.github_tools import smart_crawl_github
from src.core.context import create_context

async def test_repo():
    ctx = await create_context()
    result = await smart_crawl_github(ctx, '$repo', max_files=5, file_types_to_index=['.mdx'])
    print(f'Processed repository: {result}')

asyncio.run(test_repo())
"
done

# Content preservation validation
uv run python -c "
# Test content preservation ratio for various MDX files
# Should maintain >85% of original meaningful content
"

# Performance benchmarking
uv run python -c "
import time
# Compare MDX processing time vs Markdown processing time  
# Should be ≤ 2x markdown processing time
"

# Expected: All real-world repositories process successfully with good content preservation
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest tests/unit/features/github/processors/test_mdx_processor.py -v`
- [ ] No linting errors: `uv run ruff check src/features/github/processors/mdx_processor.py`
- [ ] No type errors: `uv run mypy src/features/github/processors/mdx_processor.py`
- [ ] No formatting issues: `uv run ruff format src/features/github/processors/mdx_processor.py --check`

### Feature Validation

- [ ] MDX files discovered during repository scanning with file_types=[".mdx"]
- [ ] JSX components detected and extracted as separate ProcessedContent entries
- [ ] Content preservation ratio >85% for cleaned text content
- [ ] Processing time ≤ 2x equivalent Markdown file processing
- [ ] Integration works with both smart_crawl_github and index_github_repository tools

### Code Quality Validation

- [ ] Follows MarkdownProcessor pattern exactly (same structure, naming, error handling)
- [ ] File placement matches existing processor organization
- [ ] Registry registration follows existing pattern in processor_factory.py
- [ ] Configuration updates follow existing settings.py patterns
- [ ] Error handling uses project exception types appropriately

### Documentation & Deployment

- [ ] Code is self-documenting with clear variable/function names
- [ ] Unicode-safe error messages (ASCII-only for Windows compatibility)  
- [ ] No new environment variables required
- [ ] Backward compatibility maintained for existing .md processing

---

## Anti-Patterns to Avoid

- ❌ Don't create new base classes - extend existing BaseFileProcessor
- ❌ Don't skip content validation - use validate_content with min_length=50
- ❌ Don't ignore frontmatter - handle with python-frontmatter library
- ❌ Don't use greedy regex - use non-greedy .*? for component content
- ❌ Don't hardcode file paths - use relative_path parameter consistently
- ❌ Don't modify existing processors - create separate MDXProcessor class
- ❌ Don't break Windows compatibility - use ASCII-only in debug output