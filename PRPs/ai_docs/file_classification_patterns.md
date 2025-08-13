# File Classification Implementation Patterns for MCP Crawl4AI RAG

## Overview

This document provides critical implementation patterns for adding intelligent file classification to the unified indexing service. These patterns ensure performance, maintainability, and integration with existing systems.

## Current Architecture Integration Points

### 1. UnifiedIndexingService Extension Pattern

**Location**: `src/services/unified_indexing_service.py`
**Key Method**: `_process_single_file()` lines 662-732

**Current Pattern**:
```python
# Process for RAG if requested
if request.should_process_rag:
    rag_success = await self._process_file_for_rag(...)

# Process for Neo4j if requested  
if request.should_process_kg:
    kg_success = await self._process_file_for_neo4j(...)
```

**Enhanced Pattern**:
```python
# NEW: Intelligent classification
classification = await self._classify_file_intelligent(file_path, content, request)

# Route based on classification instead of static request
if classification.should_process_rag:
    rag_success = await self._process_file_for_rag(...)

if classification.should_process_kg:
    kg_success = await self._process_file_for_neo4j(...)
```

### 2. Data Model Enhancement Pattern

**Location**: `src/services/unified_indexing_service.py` lines 90-112

**Current UnifiedIndexingRequest**:
```python
@dataclass
class UnifiedIndexingRequest:
    repo_url: str
    destination: IndexingDestination  # Static destination
    file_types: List[str] = field(default_factory=lambda: [".md"])
```

**Enhanced Pattern**:
```python
@dataclass
class IntelligentRoutingConfig:
    enable_intelligent_routing: bool = True
    force_rag_patterns: List[str] = field(default_factory=list)
    force_kg_patterns: List[str] = field(default_factory=list)
    classification_confidence_threshold: float = 0.8

@dataclass
class UnifiedIndexingRequest:
    repo_url: str
    destination: IndexingDestination  # Fallback when intelligent disabled
    file_types: List[str] = field(default_factory=lambda: [".md"])
    routing_config: IntelligentRoutingConfig = field(default_factory=IntelligentRoutingConfig)
```

## File Classification Algorithm Design

### 1. Fast Extension-Based Classification (Primary)

**Performance**: O(1) lookup
**Use Case**: 95% of files with clear extensions

```python
class FileClassifier:
    """Extension-based file classifier optimized for performance."""
    
    RAG_EXTENSIONS = {
        '.md', '.mdx', '.rst', '.txt', '.adoc', '.wiki',
        '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg'
    }
    
    KG_EXTENSIONS = {
        '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.scala', '.kt',
        '.go', '.rs', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
        '.cs', '.php', '.rb', '.swift', '.dart', '.m', '.mm'
    }
    
    def classify_by_extension(self, file_path: str) -> IndexingDestination:
        """O(1) classification by file extension."""
        extension = Path(file_path).suffix.lower()
        
        if extension in self.RAG_EXTENSIONS:
            return IndexingDestination.QDRANT
        elif extension in self.KG_EXTENSIONS:
            return IndexingDestination.NEO4J
        else:
            return IndexingDestination.QDRANT  # Default fallback
```

### 2. Override Pattern Integration

**Location**: Method should integrate with existing patterns
**Pattern**: Support for user-defined overrides

```python
def apply_user_overrides(self, file_path: str, base_classification: IndexingDestination, 
                        routing_config: IntelligentRoutingConfig) -> IndexingDestination:
    """Apply user-defined routing overrides."""
    import re
    
    # Check force patterns
    for pattern in routing_config.force_rag_patterns:
        if re.search(pattern, file_path, re.IGNORECASE):
            return IndexingDestination.QDRANT
            
    for pattern in routing_config.force_kg_patterns:
        if re.search(pattern, file_path, re.IGNORECASE):
            return IndexingDestination.NEO4J
    
    return base_classification
```

## Critical Performance Patterns

### 1. Caching Pattern for Classification Results

**Requirement**: Avoid reclassifying the same file types repeatedly

```python
from functools import lru_cache

class FileClassifier:
    def __init__(self):
        self._extension_cache = {}
    
    @lru_cache(maxsize=256)
    def _classify_extension_cached(self, extension: str) -> IndexingDestination:
        """Cache classification results by extension."""
        if extension in self.RAG_EXTENSIONS:
            return IndexingDestination.QDRANT
        elif extension in self.KG_EXTENSIONS:
            return IndexingDestination.NEO4J
        else:
            return IndexingDestination.QDRANT
```

### 2. Batch Processing Integration

**Location**: `src/services/unified_indexing_service.py` - `_process_files_unified()`
**Pattern**: Pre-classify files before batch processing

```python
async def _process_files_unified(self, files: List[Path], request: UnifiedIndexingRequest) -> List[FileProcessingResult]:
    """Enhanced with pre-classification for batch optimization."""
    
    # Pre-classify all files for batch optimization
    if request.routing_config.enable_intelligent_routing:
        classified_files = self._pre_classify_files(files, request)
        # Group by destination for optimized processing
        rag_files = [f for f, dest in classified_files if dest == IndexingDestination.QDRANT]
        kg_files = [f for f, dest in classified_files if dest == IndexingDestination.NEO4J]
    else:
        # Use existing logic
        pass
```

## Error Handling and Validation Patterns

### 1. Classification Confidence Pattern

```python
@dataclass
class ClassificationResult:
    destination: IndexingDestination
    confidence: float
    reasoning: str
    applied_overrides: List[str] = field(default_factory=list)
    
    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8
```

### 2. Fallback Pattern for Edge Cases

```python
def classify_with_fallback(self, file_path: str, content: Optional[str] = None) -> ClassificationResult:
    """Classify with confidence scoring and fallback logic."""
    
    try:
        # Primary: Extension-based classification
        extension_result = self._classify_by_extension(file_path)
        if extension_result.confidence > 0.9:
            return extension_result
        
        # Fallback: Default to documentation processing
        return ClassificationResult(
            destination=IndexingDestination.QDRANT,
            confidence=0.5,
            reasoning="Unknown extension, defaulting to documentation processing"
        )
    except Exception as e:
        logger.warning(f"Classification failed for {file_path}: {e}")
        return ClassificationResult(
            destination=IndexingDestination.QDRANT,
            confidence=0.0,
            reasoning=f"Classification error: {e}"
        )
```

## Integration with Existing Systems

### 1. FileProcessingResult Enhancement

**Location**: `src/services/unified_indexing_service.py`
**Pattern**: Add classification metadata without breaking changes

```python
@dataclass
class FileProcessingResult:
    # Existing fields...
    file_id: str
    processed_for_rag: bool = False
    processed_for_kg: bool = False
    
    # NEW: Classification metadata
    classification_result: Optional[ClassificationResult] = None
    routing_decision: Optional[str] = None
    classification_time_ms: float = 0.0
```

### 2. MCP Tool Parameter Enhancement

**Location**: `src/tools/github_tools.py` - `index_github_repository`
**Pattern**: Backward-compatible parameter addition

```python
async def index_github_repository(
    ctx: Context,
    repo_url: str,
    destination: str = "both",
    file_types: List[str] = None,
    max_files: int = 50,
    chunk_size: int = 5000,
    max_size_mb: int = 500,
    # NEW: Intelligent routing parameters (backward compatible)
    enable_intelligent_routing: bool = True,
    force_rag_patterns: List[str] = None,
    force_kg_patterns: List[str] = None,
) -> str:
```

## Testing Patterns

### 1. Unit Test Structure

**Location**: `tests/unit/services/test_file_classifier.py`
**Pattern**: Follow existing service test patterns

```python
class TestFileClassifier:
    """Test file classification logic."""
    
    def test_extension_classification_rag_files(self):
        """Test classification of documentation files."""
        classifier = FileClassifier()
        
        rag_files = ["README.md", "config.yaml", "docs.rst", "api.json"]
        for file_path in rag_files:
            result = classifier.classify_by_extension(file_path)
            assert result == IndexingDestination.QDRANT
    
    def test_extension_classification_kg_files(self):
        """Test classification of code files."""
        classifier = FileClassifier()
        
        kg_files = ["main.py", "index.js", "App.tsx", "service.go"]
        for file_path in kg_files:
            result = classifier.classify_by_extension(file_path)
            assert result == IndexingDestination.NEO4J
```

### 2. Integration Test Pattern

**Pattern**: Test full classification pipeline with real files

```python
@pytest.mark.asyncio
async def test_intelligent_routing_integration(mock_qdrant_client):
    """Test end-to-end intelligent routing."""
    request = UnifiedIndexingRequest(
        repo_url="https://github.com/test/repo",
        destination=IndexingDestination.BOTH,
        routing_config=IntelligentRoutingConfig(enable_intelligent_routing=True)
    )
    
    service = UnifiedIndexingService(qdrant_client=mock_qdrant_client)
    # Test with sample repository structure
    # Verify correct routing decisions
```

## Validation Commands

### 1. Classification Accuracy Validation

```bash
# Test classification accuracy on sample repository
uv run python -m src.services.file_classifier --test-repo tests/fixtures/ --validate-classification

# Expected: >95% accuracy on known file types
```

### 2. Performance Benchmark

```bash
# Benchmark classification performance
uv run python -m src.services.file_classifier --benchmark --file-count 1000

# Expected: <1ms per file for extension-based classification
```

## Critical Gotchas

1. **Threading Safety**: FileClassifier must be thread-safe for concurrent use
2. **Memory Usage**: Avoid loading file content for extension-based classification
3. **Path Normalization**: Handle Windows/Unix path differences consistently
4. **Unicode Handling**: Ensure file path handling works with international characters
5. **Performance Impact**: Classification should add <5% overhead to processing time

## Environment Configuration

```bash
# New environment variables for intelligent routing
ENABLE_INTELLIGENT_ROUTING=true
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.8
FILE_CLASSIFICATION_CACHE_SIZE=1000
```

This documentation provides the essential patterns needed for successful implementation while maintaining backward compatibility and performance standards.