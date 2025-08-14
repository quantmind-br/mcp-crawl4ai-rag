# GitHub Repository Indexing Unification - PRP Specification

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

Transform the `index_github_repository` tool to use the sophisticated content processing approach from `smart_crawl_github`, creating a unified, intelligent GitHub repository indexing system that extracts high-value content (docstrings, documentation) instead of raw file content, then deprecate the legacy `smart_crawl_github` tool.

## Mid-Level Objectives

1. **Centralize Shared Logic**: Move `smart_chunk_markdown` from web tools to utilities for reuse across the system
2. **Integrate ProcessorFactory**: Connect the existing GitHub content processors to the UnifiedIndexingService
3. **Enhance RAG Processing**: Replace raw file content indexing with intelligent content extraction using ProcessedContent
4. **Extend Documentation Support**: Add support for additional documentation formats (.rst, .txt, .adoc)
5. **Maintain Backward Compatibility**: Ensure existing Neo4j indexing and cross-system file_id linking remain functional
6. **Validate Equivalence**: Prove the enhanced system produces identical or superior results to smart_crawl_github
7. **Clean Legacy Code**: Remove deprecated smart_crawl_github function and associated tests

## Implementation Notes

### Current State Analysis

**Files Involved:**
- `src/tools/github_tools.py` - Contains both `smart_crawl_github` and `index_github_repository`
- `src/tools/web_tools.py` - Contains `smart_chunk_markdown` function 
- `src/services/unified_indexing_service.py` - Main processing service with `_process_file_for_rag` method
- `src/features/github/processors/` - Content processors (Python, Markdown, TypeScript, etc.)
- `src/features/github/core/models.py` - `ProcessedContent` data model

**Current Behavior:**
- `smart_crawl_github`: Uses ProcessorFactory to extract high-value content (docstrings, docs) 
- `index_github_repository`: Uses UnifiedIndexingService which reads raw file content for Qdrant
- Different chunking and metadata strategies between the two approaches

**Technical Debt:**
- Code duplication between indexing approaches
- `smart_chunk_markdown` located in web_tools instead of utilities
- No fallback strategy for unsupported file types in processor system

### Dependencies and Requirements

- Python 3.12+ with async/await patterns
- Existing ProcessorFactory and content processors must remain unchanged
- Qdrant client integration for vector storage
- Neo4j integration must be preserved for knowledge graph functionality
- Cross-system file_id linking between Qdrant and Neo4j

### Coding Standards

- Follow existing async patterns in UnifiedIndexingService
- Maintain comprehensive docstrings with type hints
- Use snake_case naming conventions
- Error handling with graceful degradation for processing failures
- Preserve logging and progress tracking functionality

### Risk Mitigation

- Implement processor-based logic with fallback to raw content processing
- Maintain existing batch processing and performance optimizations
- Ensure equivalence testing validates both content and metadata consistency
- Preserve all existing integration points and API signatures

## Context

### Beginning Context

**Existing Files:**
- `src/tools/web_tools.py` - Contains `smart_chunk_markdown(text: str) -> List[str]`
- `src/tools/github_tools.py` - Contains `smart_crawl_github()` and `index_github_repository()`
- `src/services/unified_indexing_service.py` - Contains `_process_file_for_rag()` with raw content processing
- `src/features/github/processors/processor_factory.py` - ProcessorFactory with registered processors
- `src/features/github/core/models.py` - ProcessedContent dataclass model

**Current Architecture:**
- Two separate GitHub indexing approaches with different content processing strategies
- ProcessorFactory exists but is not integrated with UnifiedIndexingService
- Raw file content chunking for Qdrant vs intelligent content extraction for smart_crawl_github

### Ending Context

**Modified Files:**
- `src/utils/chunking.py` - New utility module with centralized `smart_chunk_markdown`
- `src/tools/web_tools.py` - Import chunking from utils (import updated)
- `src/tools/github_tools.py` - `smart_crawl_github` function removed, `index_github_repository` preserved
- `src/services/unified_indexing_service.py` - Enhanced `_process_file_for_rag` with processor integration
- `src/features/github/processors/documentation_processor.py` - New processor for .rst, .txt, .adoc files
- `src/features/github/processors/processor_factory.py` - DocumentationProcessor registered

**New Architecture:**
- Single unified GitHub indexing approach through `index_github_repository`
- ProcessorFactory integrated with UnifiedIndexingService for intelligent content extraction
- Centralized chunking utilities shared across web and GitHub tools
- Extended documentation format support with graceful fallback processing

## Low-Level Tasks

> Ordered from start to finish with validation commands

### 1. Create Centralized Chunking Utility

**Task**: Move `smart_chunk_markdown` from web_tools to a new utils/chunking.py module

```bash
# Validation command
uv run python -c "from src.utils.chunking import smart_chunk_markdown; print('SUCCESS: Chunking utility imported')"
```

**Action**: CREATE
**File**: `src/utils/chunking.py`
**Function**: `smart_chunk_markdown(text: str, chunk_size: int = 1000) -> List[str]`
**Details**: Copy the complete function implementation from web_tools.py with all dependencies and preserve original behavior. Add proper type hints and docstring documentation.

### 2. Update Web Tools Import

**Task**: Update web_tools.py to import smart_chunk_markdown from utils

```bash
# Validation command
uv run python -c "from src.tools.web_tools import smart_chunk_markdown; print('SUCCESS: Web tools import updated')"
```

**Action**: MODIFY
**File**: `src/tools/web_tools.py`
**Function**: Import statement
**Details**: Replace the function definition with `from ..utils.chunking import smart_chunk_markdown` and verify all web tools still function correctly.

### 3. Create Documentation Processor

**Task**: Create a new processor for additional documentation formats

```bash
# Validation command
uv run python -c "from src.features.github.processors.documentation_processor import DocumentationProcessor; print('SUCCESS: Documentation processor created')"
```

**Action**: CREATE
**File**: `src/features/github/processors/documentation_processor.py`
**Function**: `DocumentationProcessor` class implementing `IFileProcessor`
**Details**: Create a simple processor that reads complete file content and returns a single ProcessedContent with content_type="documentation". Support extensions: .rst, .txt, .adoc.

### 4. Register Documentation Processor

**Task**: Add DocumentationProcessor to the ProcessorFactory registry

```bash
# Validation command
uv run python -c "from src.features.github.processors import get_default_factory; factory = get_default_factory(); print(f'Supported extensions: {factory.get_supported_extensions()}')"
```

**Action**: MODIFY
**File**: `src/features/github/processors/processor_factory.py`
**Function**: `ProcessorFactory._create_default_registry()`
**Details**: Add `registry.register("documentation", DocumentationProcessor)` to the default registry creation method.

### 5. Integrate ProcessorFactory in UnifiedIndexingService

**Task**: Add ProcessorFactory instance to UnifiedIndexingService initialization

```bash
# Validation command
uv run python -c "from src.services.unified_indexing_service import UnifiedIndexingService; service = UnifiedIndexingService(None, None, None); print('SUCCESS: Service initialization with ProcessorFactory')"
```

**Action**: MODIFY
**File**: `src/services/unified_indexing_service.py`
**Function**: `UnifiedIndexingService.__init__()`
**Details**: Add import for ProcessorFactory and instantiate `self.processor_factory = ProcessorFactory()` in the initialization method.

### 6. Transform RAG Processing Logic

**Task**: Replace raw content processing with intelligent processor-based extraction

```bash
# Validation command
uv run pytest tests/unit/services/test_unified_indexing_service.py::test_process_file_for_rag_with_processors -v
```

**Action**: MODIFY
**File**: `src/services/unified_indexing_service.py`
**Function**: `_process_file_for_rag()`
**Details**: 
- Import chunking utility from utils
- Add processor discovery: `processor = self.processor_factory.get_processor_for_file(str(file_path))`
- Implement conditional logic: if processor exists, extract ProcessedContent items and chunk each item's content; if no processor, fallback to current raw content chunking
- Enhance metadata with ProcessedContent attributes (content_type, name, signature, line_number)
- Preserve file_id linking and all existing functionality

### 7. Add Helper Method for Content Chunking

**Task**: Create a reusable method for chunking ProcessedContent in UnifiedIndexingService

```bash
# Validation command
uv run python -c "from src.services.unified_indexing_service import UnifiedIndexingService; print('SUCCESS: Chunking method available')"
```

**Action**: MODIFY
**File**: `src/services/unified_indexing_service.py`
**Function**: `_chunk_processed_content()` (new method)
**Details**: Create method that takes ProcessedContent and chunk_size, returns List[str] using the centralized smart_chunk_markdown function. This enables clean separation between content processing and chunking.

### 8. Create Equivalence Test Suite

**Task**: Develop comprehensive tests to validate the migration maintains functionality

```bash
# Validation command
uv run pytest tests/integration/test_github_indexing_equivalence.py -v
```

**Action**: CREATE
**File**: `tests/integration/test_github_indexing_equivalence.py`
**Function**: Test suite with `test_processor_vs_raw_content()`, `test_metadata_consistency()`, `test_chunking_equivalence()`
**Details**: Create tests that compare output between old smart_crawl_github approach and new enhanced index_github_repository. Validate content chunks, metadata structure, and Qdrant storage results are equivalent or improved.

### 9. Update GitHub Tools Imports

**Task**: Update import statements in github_tools.py for chunking utility

```bash
# Validation command
uv run python -c "from src.tools.github_tools import index_github_repository; print('SUCCESS: GitHub tools updated')"
```

**Action**: MODIFY
**File**: `src/tools/github_tools.py`
**Function**: Import statements
**Details**: Add `from ..utils.chunking import smart_chunk_markdown` import and verify any remaining references to chunking functionality work correctly.

### 10. Comprehensive Integration Testing

**Task**: Run full test suite to verify no regressions in existing functionality

```bash
# Validation command
uv run pytest tests/unit/services/ tests/unit/tools/ tests/integration/ -v
```

**Action**: VALIDATE
**File**: All affected test files
**Function**: Complete test validation
**Details**: Execute comprehensive test suite covering UnifiedIndexingService, GitHub tools, web tools, and integration scenarios. Verify all tests pass and no functionality is broken.

### 11. Remove Legacy smart_crawl_github Function

**Task**: Delete the deprecated smart_crawl_github function and its tests

```bash
# Validation command
uv run python -c "import src.tools.github_tools as gt; assert not hasattr(gt, 'smart_crawl_github'), 'smart_crawl_github still exists'; print('SUCCESS: Legacy function removed')"
```

**Action**: DELETE
**File**: `src/tools/github_tools.py`
**Function**: `smart_crawl_github()` function
**Details**: Remove the complete function definition and any imports specific to it. Ensure index_github_repository is the sole GitHub indexing tool.

### 12. Clean Legacy Tests

**Task**: Remove tests specific to smart_crawl_github that are now redundant

```bash
# Validation command
uv run pytest tests/ -k "not smart_crawl_github" -v
```

**Action**: DELETE
**File**: `tests/unit/tools/test_github_tools.py` (specific test methods)
**Function**: Tests methods for `smart_crawl_github`
**Details**: Remove test methods that specifically tested smart_crawl_github functionality, keeping only index_github_repository tests and the new equivalence tests.

### 13. Final Validation and Documentation Update

**Task**: Verify complete system functionality and update relevant documentation

```bash
# Validation commands
uv run -m src  # Verify server starts successfully
uv run pytest  # Verify all tests pass
uv run ruff check .  # Verify code quality
```

**Action**: VALIDATE + MODIFY
**File**: Documentation files (CLAUDE.md, README.md if needed)
**Function**: System validation
**Details**: Ensure the MCP server starts successfully, all tests pass, code quality checks pass, and update any documentation that referenced smart_crawl_github to point to the unified index_github_repository approach.

## Success Criteria

1. **Functional Equivalence**: New system produces identical or superior Qdrant indexing results compared to smart_crawl_github
2. **Enhanced Capability**: Support for additional documentation formats (.rst, .txt, .adoc)
3. **Code Quality**: All tests pass, no linting errors, proper type hints and documentation
4. **Performance**: No regression in processing speed or memory usage
5. **Compatibility**: Existing Neo4j indexing and file_id linking remain fully functional
6. **Cleanliness**: Legacy smart_crawl_github code completely removed
7. **Maintainability**: Centralized chunking logic, clear separation of concerns, unified architecture

## Rollback Strategy

If critical issues arise during implementation:

1. **Immediate Rollback**: Revert changes to `unified_indexing_service.py` to restore raw content processing
2. **Restore Imports**: Revert web_tools.py import changes to restore original smart_chunk_markdown location  
3. **Re-enable Legacy**: Temporarily restore smart_crawl_github function if needed for compatibility
4. **Validation**: Run full test suite to ensure rollback success

The modular approach ensures each step can be individually reverted without affecting the entire system.