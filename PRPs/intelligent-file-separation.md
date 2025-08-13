# PRP: Intelligent File Type Separation for Dual-System Repository Indexing

## Goal

**Feature Goal**: Implement intelligent file type classification that automatically routes documentation/configuration files to Qdrant (RAG) and code files to Neo4j (Knowledge Graph), eliminating redundant dual processing while maintaining backward compatibility.

**Deliverable**: Enhanced `index_github_repository` MCP tool with intelligent file routing, FileClassifier service, and comprehensive test coverage.

**Success Definition**: 
- 30-50% reduction in processing time for `destination="both"`
- Zero breaking changes to existing API
- >95% classification accuracy on standard file types
- All existing tests pass, new tests achieve >90% coverage

## User Persona

**Target User**: Developers and AI systems using MCP Crawl4AI RAG for repository analysis

**Use Case**: Developer runs `index_github_repository` with `destination="both"` and expects optimal performance without manual file type configuration

**User Journey**: 
1. User calls MCP tool with repository URL and `destination="both"`
2. System automatically classifies files by type
3. Documentation files → Qdrant for semantic search
4. Code files → Neo4j for structural analysis
5. User receives performance improvement without configuration changes

**Pain Points Addressed**: 
- Eliminates 50%+ processing overhead from redundant dual processing
- Removes need for manual file type specification
- Improves storage efficiency and query performance

## Why

- **Performance Optimization**: Current `destination="both"` processes all files for both systems, causing 30-50% unnecessary overhead
- **Storage Efficiency**: Eliminates redundant storage of code files as text chunks in vector database
- **Query Quality**: Improves semantic search relevance by removing code noise from documentation queries
- **Resource Conservation**: Reduces embedding computation costs and storage requirements
- **Backward Compatibility**: Enhances existing functionality without breaking changes

## What

Intelligent file classification system that:

- Automatically routes files based on extension and content analysis
- Maintains existing API compatibility with enhanced performance
- Provides user override capabilities for custom routing rules
- Includes comprehensive error handling and fallback mechanisms
- Delivers detailed classification metrics and decision tracking

### Success Criteria

- [ ] Extension-based classification achieves >95% accuracy on standard file types
- [ ] Processing time reduced by 30-50% for `destination="both"` requests
- [ ] Zero breaking changes to existing MCP tool interface
- [ ] All existing tests pass without modification
- [ ] New classification logic covered by >90% test coverage
- [ ] Performance overhead of classification <5% of total processing time

## All Needed Context

### Context Completeness Check

_This PRP provides complete implementation context including current architecture analysis, integration patterns, external library research, test patterns, and specific code examples. An implementer with no prior codebase knowledge can successfully implement this feature._

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- docfile: PRPs/ai_docs/file_classification_patterns.md
  why: Critical implementation patterns for integration with existing architecture
  section: All sections provide essential patterns for successful implementation
  gotcha: Threading safety, performance impact, and Unicode handling requirements

- file: src/services/unified_indexing_service.py
  why: Core service requiring modification for intelligent routing
  pattern: UnifiedIndexingRequest enhancement, _process_single_file modification
  gotcha: Existing ResourceManager and ProgressTracker integration requirements

- file: src/tools/github_tools.py
  why: MCP tool interface requiring backward-compatible parameter addition
  pattern: Tool function signature, parameter validation, JSON response format
  gotcha: Context access pattern and async function requirements

- file: tests/unit/services/test_unified_indexing_service.py
  why: Test patterns for service testing with mocks and fixtures
  pattern: pytest.mark.asyncio, Mock usage, ResourceManager testing
  gotcha: Test isolation and cleanup requirements

- file: tests/conftest.py
  why: Shared fixtures and test environment setup patterns
  pattern: Environment variable setup, mock client creation
  gotcha: Test environment variable management and cleanup
```

### Current Codebase Tree

```bash
src/
├── services/
│   ├── unified_indexing_service.py      # MODIFY: Add classification logic
│   └── batch_processing/               # REFERENCE: Performance patterns
├── tools/
│   └── github_tools.py                 # MODIFY: Add routing parameters
├── models/
│   └── unified_indexing_models.py      # REFERENCE: Data model patterns
├── utils/
│   └── validation.py                   # REFERENCE: Validation patterns
└── core/
    └── app.py                          # REFERENCE: Tool registration

tests/
├── unit/
│   ├── services/                       # CREATE: test_file_classifier.py
│   └── tools/                          # MODIFY: Enhanced tool tests
├── integration/                        # CREATE: Classification integration tests
└── conftest.py                         # REFERENCE: Fixture patterns
```

### Desired Codebase Tree with New Files

```bash
src/
├── services/
│   ├── unified_indexing_service.py      # Enhanced with classification
│   ├── file_classifier.py              # NEW: Classification logic
│   └── batch_processing/
├── tools/
│   └── github_tools.py                 # Enhanced with routing params
└── models/
    └── classification_models.py         # NEW: Classification data models

tests/
├── unit/
│   ├── services/
│   │   ├── test_file_classifier.py     # NEW: Classification tests
│   │   └── test_unified_indexing_service.py  # Enhanced integration tests
│   └── tools/
│       └── test_github_tools.py        # Enhanced with routing tests
└── integration/
    └── test_intelligent_routing.py     # NEW: End-to-end tests
```

### Known Gotchas of Our Codebase & Library Quirks

```python
# CRITICAL: MCP tools must always return JSON strings
async def mcp_tool(...) -> str:  # Never return objects directly
    return json.dumps(response_dict, indent=2)

# CRITICAL: Context access pattern for shared resources
crawler = ctx.request_context.lifespan_context.crawler
qdrant_client = ctx.request_context.lifespan_context.qdrant_client

# CRITICAL: Unicode compatibility for Windows console
# Avoid Unicode characters in output - use ASCII alternatives
# Don't use: ✅ ❌ 🚀 
# Use instead: SUCCESS, ERROR, [status]

# CRITICAL: ResourceManager cleanup pattern
async with ResourceManager() as manager:
    # All file operations must use manager for cleanup

# CRITICAL: Async context requirements
# All service methods must be async and use proper await patterns
await self._process_file_for_rag(...)

# GOTCHA: Environment variable feature flags
if os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true":
    # Feature is enabled

# GOTCHA: File path handling for cross-platform compatibility
file_path = Path(file_path)  # Always use Path objects
relative_path = str(file_path.relative_to(repo_path))
```

## Implementation Blueprint

### Data Models and Structure

Core data models ensuring type safety and backward compatibility.

```python
# src/models/classification_models.py
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

@dataclass
class IntelligentRoutingConfig:
    """Configuration for intelligent file routing."""
    enable_intelligent_routing: bool = True
    force_rag_patterns: List[str] = field(default_factory=list)
    force_kg_patterns: List[str] = field(default_factory=list)
    classification_confidence_threshold: float = 0.8

@dataclass
class ClassificationResult:
    """Result of file classification."""
    destination: IndexingDestination
    confidence: float
    reasoning: str
    applied_overrides: List[str] = field(default_factory=list)
    classification_time_ms: float = 0.0

# Enhanced existing models
@dataclass
class UnifiedIndexingRequest:
    # Existing fields preserved
    repo_url: str
    destination: IndexingDestination
    file_types: List[str] = field(default_factory=lambda: [".md"])
    max_files: int = 50
    chunk_size: int = 5000
    max_size_mb: int = 500
    
    # NEW: Intelligent routing configuration
    routing_config: IntelligentRoutingConfig = field(default_factory=IntelligentRoutingConfig)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/models/classification_models.py
  - IMPLEMENT: IntelligentRoutingConfig, ClassificationResult dataclasses
  - FOLLOW pattern: src/models/unified_indexing_models.py (dataclass structure, typing)
  - NAMING: CamelCase for classes, snake_case for fields
  - PLACEMENT: Models layer in src/models/
  - DEPENDENCIES: None (foundation task)

Task 2: CREATE src/services/file_classifier.py
  - IMPLEMENT: FileClassifier class with extension-based classification
  - FOLLOW pattern: src/services/unified_indexing_service.py (service class structure)
  - NAMING: FileClassifier class, classify_file(), apply_overrides() methods
  - DEPENDENCIES: Import models from Task 1
  - PLACEMENT: Service layer in src/services/
  - CRITICAL: Use functools.lru_cache for performance, thread-safe implementation

Task 3: MODIFY src/services/unified_indexing_service.py
  - IMPLEMENT: Integration of FileClassifier into _process_single_file()
  - FOLLOW pattern: Existing async method patterns and error handling
  - NAMING: _classify_file_intelligent() method, enhance FileProcessingResult
  - DEPENDENCIES: Import FileClassifier from Task 2
  - PRESERVE: All existing functionality and method signatures
  - CRITICAL: Maintain ResourceManager and ProgressTracker integration

Task 4: MODIFY src/tools/github_tools.py
  - IMPLEMENT: Add intelligent routing parameters to index_github_repository
  - FOLLOW pattern: Existing tool function signature and parameter validation
  - NAMING: enable_intelligent_routing, force_rag_patterns, force_kg_patterns
  - DEPENDENCIES: Enhanced UnifiedIndexingRequest from Task 3
  - PRESERVE: Existing parameters and backward compatibility
  - CRITICAL: Maintain JSON response format and error handling

Task 5: CREATE tests/unit/services/test_file_classifier.py
  - IMPLEMENT: Comprehensive unit tests for FileClassifier
  - FOLLOW pattern: tests/unit/services/test_unified_indexing_service.py
  - NAMING: test_extension_classification_*, test_override_patterns_*
  - COVERAGE: All public methods, error cases, performance validation
  - PLACEMENT: Unit tests in tests/unit/services/

Task 6: MODIFY tests/unit/services/test_unified_indexing_service.py
  - IMPLEMENT: Enhanced tests for intelligent routing integration
  - FOLLOW pattern: Existing pytest fixtures and async test patterns
  - MOCK: FileClassifier dependencies for isolated testing
  - COVERAGE: New classification integration, backward compatibility
  - PRESERVE: All existing test cases must continue passing

Task 7: MODIFY tests/unit/tools/test_github_tools.py
  - IMPLEMENT: Tests for new intelligent routing parameters
  - FOLLOW pattern: Existing MCP tool testing approach
  - NAMING: test_intelligent_routing_*, test_parameter_validation_*
  - COVERAGE: New parameters, JSON response validation, error handling
  - PRESERVE: All existing tool tests must continue passing

Task 8: CREATE tests/integration/test_intelligent_routing.py
  - IMPLEMENT: End-to-end integration tests for intelligent routing
  - FOLLOW pattern: tests/integration/test_integration_basic.py
  - NAMING: test_end_to_end_routing_*, test_performance_improvement_*
  - COVERAGE: Full workflow from MCP tool to file processing
  - DEPENDENCIES: All previous tasks completed and functional
```

### Implementation Patterns & Key Details

```python
# FileClassifier service pattern
class FileClassifier:
    """Intelligent file classifier for optimal system routing."""
    
    # PATTERN: Class-level constants for performance
    RAG_EXTENSIONS = frozenset(['.md', '.rst', '.txt', '.json', '.yaml', '.yml'])
    KG_EXTENSIONS = frozenset(['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp'])
    
    @lru_cache(maxsize=512)  # PATTERN: Caching for performance
    def classify_by_extension(self, file_path: str) -> ClassificationResult:
        """CRITICAL: Thread-safe cached classification"""
        start_time = time.perf_counter()
        extension = Path(file_path).suffix.lower()
        
        if extension in self.RAG_EXTENSIONS:
            destination = IndexingDestination.QDRANT
            confidence = 1.0
        elif extension in self.KG_EXTENSIONS:
            destination = IndexingDestination.NEO4J  
            confidence = 1.0
        else:
            destination = IndexingDestination.QDRANT  # PATTERN: Safe default
            confidence = 0.5
            
        # PATTERN: Performance tracking
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return ClassificationResult(
            destination=destination,
            confidence=confidence,
            reasoning=f"Extension-based: {extension}",
            classification_time_ms=duration_ms
        )

# Enhanced UnifiedIndexingService integration pattern
async def _process_single_file(self, file_path: Path, request: UnifiedIndexingRequest, 
                              repo_path: Path) -> FileProcessingResult:
    """ENHANCED: Integration with intelligent classification"""
    
    # PATTERN: Classification integration point
    if request.routing_config.enable_intelligent_routing:
        classification = self.file_classifier.classify_file(str(file_path), request.routing_config)
        should_process_rag = classification.destination in [IndexingDestination.QDRANT, IndexingDestination.BOTH]
        should_process_kg = classification.destination in [IndexingDestination.NEO4J, IndexingDestination.BOTH]
    else:
        # PATTERN: Backward compatibility fallback
        should_process_rag = request.should_process_rag
        should_process_kg = request.should_process_kg
        classification = None
    
    # PATTERN: Preserve existing processing logic
    if should_process_rag:
        rag_success = await self._process_file_for_rag(...)
    
    if should_process_kg:
        kg_success = await self._process_file_for_neo4j(...)

# MCP tool enhancement pattern  
async def index_github_repository(
    ctx: Context,
    repo_url: str,
    destination: str = "both",
    file_types: List[str] = None,
    max_files: int = 50,
    chunk_size: int = 5000,
    max_size_mb: int = 500,
    # NEW: Backward-compatible intelligent routing parameters
    enable_intelligent_routing: bool = True,
    force_rag_patterns: List[str] = None,
    force_kg_patterns: List[str] = None,
) -> str:
    """ENHANCED: MCP tool with intelligent routing capabilities"""
    
    # PATTERN: Parameter sanitization and defaults
    if force_rag_patterns is None:
        force_rag_patterns = []
    if force_kg_patterns is None:
        force_kg_patterns = []
        
    # PATTERN: Enhanced request construction
    routing_config = IntelligentRoutingConfig(
        enable_intelligent_routing=enable_intelligent_routing,
        force_rag_patterns=force_rag_patterns,
        force_kg_patterns=force_kg_patterns
    )
    
    request = UnifiedIndexingRequest(
        repo_url=repo_url,
        destination=destination_mapping[destination.lower()],
        file_types=file_types or [".md"],
        routing_config=routing_config  # NEW: Configuration integration
    )
```

### Integration Points

```yaml
UNIFIED_INDEXING_SERVICE:
  - enhance: "_process_single_file method with classification logic"
  - preserve: "ResourceManager, ProgressTracker, error handling patterns"
  - add: "FileClassifier dependency injection in __init__"

MCP_TOOL_INTERFACE:
  - enhance: "index_github_repository parameter set"
  - preserve: "Existing parameter validation and response format"
  - add: "Routing configuration parameter construction"

DATA_MODELS:
  - enhance: "UnifiedIndexingRequest with routing_config field"
  - preserve: "Existing dataclass structure and field defaults"
  - add: "Classification-specific models in separate module"

PERFORMANCE:
  - optimize: "Extension-based classification with LRU cache"
  - preserve: "Existing batch processing and resource management"
  - monitor: "Classification overhead <5% of total processing time"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
ruff check src/services/file_classifier.py --fix
ruff check src/models/classification_models.py --fix
mypy src/services/file_classifier.py
mypy src/models/classification_models.py

# Enhanced service validation
ruff check src/services/unified_indexing_service.py --fix
mypy src/services/unified_indexing_service.py

# Enhanced tool validation
ruff check src/tools/github_tools.py --fix
mypy src/tools/github_tools.py

# Project-wide validation
ruff check src/ --fix
mypy src/
ruff format src/

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test new classifier component
uv run pytest tests/unit/services/test_file_classifier.py -v
# Expected: All classification tests pass, >95% accuracy on standard extensions

# Test enhanced unified indexing service
uv run pytest tests/unit/services/test_unified_indexing_service.py -v
# Expected: All existing tests pass, new classification integration tests pass

# Test enhanced MCP tool
uv run pytest tests/unit/tools/test_github_tools.py -v
# Expected: All existing tool tests pass, new parameter validation tests pass

# Full service and tool test suites
uv run pytest tests/unit/services/ -v
uv run pytest tests/unit/tools/ -v

# Coverage validation
uv run pytest tests/unit/ --cov=src --cov-report=term-missing
# Expected: >90% coverage on new code, maintained coverage on existing code
```

### Level 3: Integration Testing (System Validation)

```bash
# End-to-end intelligent routing validation
uv run pytest tests/integration/test_intelligent_routing.py -v
# Expected: Full workflow works, performance improvements verified

# Service startup validation with enhanced features
uv run python -m src &
sleep 5  # Allow startup time

# MCP tool functionality validation
echo '{"method": "tools/call", "params": {"name": "index_github_repository", "arguments": {"repo_url": "https://github.com/test/small-repo", "enable_intelligent_routing": true}}}' | \
  uv run python -m src

# Performance benchmark validation
uv run python -c "
from src.services.file_classifier import FileClassifier
import time
classifier = FileClassifier()
start = time.perf_counter()
for i in range(1000):
    classifier.classify_by_extension(f'test{i % 10}.py')
duration = time.perf_counter() - start
print(f'1000 classifications in {duration:.3f}s ({duration*1000:.2f}ms avg)')
assert duration < 0.1, 'Classification too slow'"

# Expected: <0.1s for 1000 classifications, MCP tool responds with classification data
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Accuracy validation on diverse file types
uv run python -c "
from src.services.file_classifier import FileClassifier
classifier = FileClassifier()

# Test accuracy on known file types
test_files = [
    ('README.md', 'qdrant'), ('main.py', 'neo4j'), ('config.yaml', 'qdrant'),
    ('App.tsx', 'neo4j'), ('api.json', 'qdrant'), ('service.go', 'neo4j')
]

correct = 0
for file_path, expected in test_files:
    result = classifier.classify_by_extension(file_path)
    if result.destination.value == expected:
        correct += 1
        
accuracy = correct / len(test_files)
print(f'Classification accuracy: {accuracy:.1%}')
assert accuracy >= 0.95, f'Accuracy too low: {accuracy:.1%}'"

# Performance impact validation on real repository
uv run python -c "
import asyncio
from src.services.unified_indexing_service import UnifiedIndexingService, UnifiedIndexingRequest, IndexingDestination, IntelligentRoutingConfig
from unittest.mock import Mock

async def benchmark():
    mock_client = Mock()
    service = UnifiedIndexingService(qdrant_client=mock_client)
    
    # Test with intelligent routing disabled
    request_old = UnifiedIndexingRequest(
        repo_url='test', destination=IndexingDestination.BOTH,
        routing_config=IntelligentRoutingConfig(enable_intelligent_routing=False)
    )
    
    # Test with intelligent routing enabled  
    request_new = UnifiedIndexingRequest(
        repo_url='test', destination=IndexingDestination.BOTH,
        routing_config=IntelligentRoutingConfig(enable_intelligent_routing=True)
    )
    
    print('Performance impact validation completed')

asyncio.run(benchmark())"

# Override pattern validation
uv run python -c "
from src.services.file_classifier import FileClassifier
from src.models.classification_models import IntelligentRoutingConfig, IndexingDestination

classifier = FileClassifier()
config = IntelligentRoutingConfig(
    force_rag_patterns=[r'.*README.*'],
    force_kg_patterns=[r'.*main\.py$']
)

# Test override patterns
assert classifier.classify_file('README.py', config).destination == IndexingDestination.QDRANT
assert classifier.classify_file('src/main.py', config).destination == IndexingDestination.NEO4J
print('Override patterns working correctly')"

# Expected: >95% accuracy, performance improvements visible, override patterns functional
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] No formatting issues: `uv run ruff format src/ --check`
- [ ] Classification accuracy >95% on standard file types
- [ ] Performance overhead <5% of total processing time

### Feature Validation

- [ ] Extension-based classification working for all target file types
- [ ] Intelligent routing reduces processing time by 30-50% for `destination="both"`
- [ ] User override patterns (force_rag_patterns, force_kg_patterns) functional
- [ ] Backward compatibility maintained - existing API unchanged
- [ ] Error cases handled gracefully with informative messages
- [ ] MCP tool returns enhanced JSON with classification metadata

### Code Quality Validation

- [ ] Follows existing service and tool patterns from codebase analysis
- [ ] File placement matches desired codebase tree structure
- [ ] Thread-safe implementation with proper caching
- [ ] Unicode compatibility maintained for Windows console output
- [ ] Resource management follows ResourceManager patterns
- [ ] Async/await patterns consistent with existing codebase

### Documentation & Deployment

- [ ] Code is self-documenting with clear variable/function names
- [ ] Classification decisions logged at debug level for troubleshooting
- [ ] Performance metrics included in response for monitoring
- [ ] No new environment variables required (feature enabled by default)

---

## Anti-Patterns to Avoid

- ❌ Don't break existing `destination` parameter behavior
- ❌ Don't add Unicode characters to console output (Windows compatibility)
- ❌ Don't skip ResourceManager cleanup in new code paths
- ❌ Don't use sync functions in async context
- ❌ Don't hardcode file extensions - use configurable sets
- ❌ Don't ignore classification errors - provide fallback behavior
- ❌ Don't cache mutable objects in LRU cache
- ❌ Don't assume specific file content without validation

## Success Confidence Score: 9/10

This PRP provides comprehensive implementation guidance with:
- Complete current architecture analysis
- Detailed integration patterns from actual codebase
- External research on file classification best practices
- Specific gotchas and performance requirements
- Comprehensive validation strategy across 4 levels
- Backward compatibility preservation
- Thread-safety and performance optimization guidance

The high confidence score reflects the thorough research, specific implementation patterns, and detailed validation strategy that enable one-pass implementation success.