# DeepInfra Qwen3-Embedding-0.6B Integration PRP

## Goal

Integrate DeepInfra's Qwen3-Embedding-0.6B model (1024 dimensions) with automatic dimension validation and table recreation. The application must:

1. Support DeepInfra API configuration via environment variables
2. Add `EMBEDDINGS_DIMENSIONS` configuration to .env with intelligent defaults
3. Automatically validate Qdrant collection dimensions on startup
4. Recreate collections when dimensions don't match configured model
5. Maintain backward compatibility with existing OpenAI configurations

## Why

- **Cost Optimization**: DeepInfra offers competitive pricing at $0.002 per million tokens
- **Multilingual Support**: Qwen3-Embedding-0.6B supports 100+ languages with excellent performance
- **Flexibility**: Support multiple embedding providers and models with varying dimensions
- **Future-Proofing**: Dynamic dimension system enables easy model switching
- **Performance**: 1024 dimensions may provide sufficient quality with faster processing

## What

### User-Visible Behavior
- Application starts successfully with DeepInfra configuration
- Automatic detection and resolution of dimension mismatches
- Clear logging when collections are recreated
- Seamless switching between embedding providers
- Backward compatibility with existing installations

### Technical Requirements
- Dynamic dimension detection based on embedding model
- Qdrant collection validation and recreation logic
- Environment variable configuration for dimensions
- Error handling for API failures and misconfigurations
- Comprehensive test coverage for all scenarios

### Success Criteria
- [ ] Application starts with DeepInfra Qwen3 configuration and creates 1024-dimension collections
- [ ] Existing installations with 1536-dimension collections are automatically migrated
- [ ] OpenAI configurations continue working without changes
- [ ] All tests pass with new dimension system
- [ ] Lint and type checking pass
- [ ] Integration tests validate end-to-end functionality

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Critical for implementation
- url: https://deepinfra.com/docs/openai_api
  why: OpenAI-compatible API endpoints and request format
  critical: Use /v1/openai/embeddings endpoint for compatibility

- url: https://deepinfra.com/docs/api-reference  
  why: Authentication, rate limits, and error handling patterns
  critical: 429 errors at >200 concurrent requests, need proper backoff

- url: https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B
  why: Model specifications - 1024 max dimensions, 32K context window
  critical: Supports dimensions 32-1024, defaults to model's native size

- file: src/utils.py:391-445
  why: Current embedding creation patterns and retry logic to preserve
  critical: Batch processing with individual fallback on errors

- file: src/qdrant_wrapper.py:27-51  
  why: Current collection schema - hardcoded 1536 dimensions to replace
  critical: COLLECTIONS dict structure must be preserved for compatibility

- file: tests/test_utils_integration.py
  why: Test patterns for embedding functionality and mocking
  critical: Follow existing mock patterns for OpenAI client

- file: .env.example:15-22
  why: Current environment variable organization and naming conventions
  critical: Follow EMBEDDINGS_* prefix pattern for consistency
```

### Current Codebase Tree

```bash
E:\mcp-crawl4ai-rag\
├── src/
│   ├── utils.py                     # Embedding utilities (MODIFY)
│   ├── qdrant_wrapper.py           # Vector DB integration (MODIFY)
│   └── crawl4ai_mcp.py             # MCP server startup (MODIFY)
├── tests/
│   ├── conftest.py                 # Test configuration (MODIFY)
│   ├── test_utils_integration.py   # Embedding tests (MODIFY)
│   └── test_qdrant_wrapper.py      # Vector DB tests (MODIFY)
├── .env.example                    # Environment template (MODIFY)
└── PRPs/
    └── deepinfra-qwen3-embedding-integration.md
```

### Desired Codebase Tree

```bash
# No new files needed - only modifications to existing files
# All changes integrate into existing architecture
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: Qdrant dimension changes require collection recreation
# - Cannot modify vector dimensions of existing collections
# - Must delete and recreate collections for dimension changes
# - All data in collections will be lost during recreation

# CRITICAL: DeepInfra Qwen3 model constraints
# - Maximum 1024 dimensions (not unlimited)
# - Dimensions must be between 32-1024
# - Default dimension varies by model

# CRITICAL: Current hardcoded dimension locations
# src/qdrant_wrapper.py:29,40 - VectorParams(size=1536)
# src/utils.py:442,459,463 - [0.0] * 1536 fallback vectors
# Multiple test files - Mock embeddings with 1536 dimensions

# CRITICAL: OpenAI API compatibility
# - DeepInfra uses /v1/openai/embeddings endpoint
# - Response format identical to OpenAI
# - Rate limits: 200 concurrent requests max

# CRITICAL: Environment variable loading
# - Uses python-dotenv with override=True
# - Variables loaded in crawl4ai_mcp.py startup
# - Fallback pattern: os.getenv("VAR", "default")
```

## Implementation Blueprint

### Data Models and Structure

The embedding system uses these core components:
- **Embedding Client**: OpenAI-compatible client for API calls
- **Vector Collections**: Qdrant collections with dimension-specific configuration  
- **Configuration System**: Environment-based configuration with intelligent defaults

```python
# Core dimension management structure
@dataclass
class EmbeddingConfig:
    model: str
    dimensions: int
    api_key: str
    base_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        # Auto-detect dimensions from model or explicit config
        pass
```

### List of Tasks (Implementation Order)

```yaml
Task 1 - Add Dynamic Dimension Utilities:
MODIFY src/utils.py:
  - ADD get_embedding_dimensions() function after line 86
  - ADD validate_embeddings_config() function for validation
  - REPLACE hardcoded 1536 dimensions in fallback vectors (lines 442, 459, 463)
  - PRESERVE existing create_embeddings_batch() function signature

Task 2 - Update Environment Configuration:
MODIFY .env.example:
  - ADD EMBEDDINGS_DIMENSIONS variable in AI Models section after line 22
  - ADD comment explaining dimension configuration and model examples
  - PRESERVE existing variable organization and naming conventions

Task 3 - Implement Dynamic Collection Configuration:
MODIFY src/qdrant_wrapper.py:
  - REPLACE global COLLECTIONS dict with get_collections_config() function
  - ADD dimension validation methods to QdrantClientWrapper class
  - MODIFY _ensure_collections_exist() to include dimension validation
  - PRESERVE existing collection schema structure and field names

Task 4 - Add Startup Dimension Validation:
MODIFY src/crawl4ai_mcp.py:
  - ADD dimension validation call during startup initialization
  - INTEGRATE with existing error handling patterns
  - ADD informative logging for dimension changes and collection recreation
  - PRESERVE existing startup sequence and error handling

Task 5 - Update Test Infrastructure:
MODIFY tests/conftest.py:
  - ADD EMBEDDINGS_DIMENSIONS to test environment setup
  - UPDATE mock embedding dimensions to use dynamic values
  - PRESERVE existing test environment patterns

CREATE tests/test_deepinfra_integration.py:
  - ADD comprehensive DeepInfra configuration tests
  - ADD dimension validation and migration tests
  - FOLLOW existing test patterns and naming conventions

MODIFY tests/test_utils_integration.py:
  - UPDATE hardcoded 1536 dimensions to use dynamic values
  - ADD tests for new dimension utility functions
  - PRESERVE existing test structure and mock patterns

MODIFY tests/test_qdrant_wrapper.py:
  - UPDATE collection tests to use dynamic dimensions
  - ADD dimension validation and recreation tests
  - PRESERVE existing QdrantClientWrapper test patterns
```

### Per Task Pseudocode

```python
# Task 1 - Dynamic Dimension Utilities
def get_embedding_dimensions() -> int:
    """Get embedding dimensions with model-based defaults."""
    # Check explicit EMBEDDINGS_DIMENSIONS first
    explicit_dims = os.getenv("EMBEDDINGS_DIMENSIONS")
    if explicit_dims:
        return int(explicit_dims)  # Validate and return
    
    # Auto-detect from model name
    model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    model_dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072, 
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        # Add more models as needed
    }
    
    return model_dimensions.get(model, 1536)  # Default fallback

# Task 3 - Dynamic Collections
def get_collections_config():
    """Generate collection config with current dimensions."""
    dims = get_embedding_dimensions()
    return {
        "crawled_pages": {
            "vectors_config": VectorParams(size=dims, distance=Distance.COSINE),
            # Preserve existing payload schema
        }
    }

def _validate_collection_dimensions(self, collection_name, expected_config):
    """Check if collection dimensions match expected config."""
    try:
        info = self.client.get_collection(collection_name)
        current_size = info.config.params.vectors.size
        expected_size = expected_config.size
        return {
            "needs_recreation": current_size != expected_size,
            "current_size": current_size,
            "expected_size": expected_size
        }
    except Exception:
        return {"needs_recreation": True}

# Task 4 - Startup Validation Integration
async def validate_embeddings_setup():
    """Validate embedding configuration on startup."""
    # Validate environment configuration
    validate_embeddings_config()
    
    # Initialize Qdrant with dimension validation
    client = get_qdrant_client()
    # Client initialization triggers collection validation automatically
    
    logger.info(f"Embedding setup validated - dimensions: {get_embedding_dimensions()}")
```

### Integration Points

```yaml
ENVIRONMENT:
  - add to: .env.example after line 22
  - pattern: "EMBEDDINGS_DIMENSIONS=1024  # DeepInfra Qwen3 model"
  - validation: Must be positive integer, reasonable upper limit

QDRANT:
  - modify: Collection creation in QdrantClientWrapper._ensure_collections_exist()
  - pattern: Validate dimensions before creating collections
  - migration: Delete and recreate collections on dimension mismatch

API_COMPATIBILITY:
  - preserve: OpenAI-compatible interface in utils.py
  - endpoint: DeepInfra uses /v1/openai/embeddings (already compatible)
  - authentication: Bearer token in Authorization header (already supported)

TESTING:
  - pattern: Follow conftest.py environment setup for new variables
  - mocking: Update embedding mocks to use dynamic dimensions
  - integration: Test dimension validation and collection recreation
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix                 # Auto-fix formatting and style issues
mypy src/                             # Type checking validation

# Expected: No errors. If errors occur, read error messages and fix code.
```

### Level 2: Unit Tests

```python
# Key test cases for each modified component:

def test_get_embedding_dimensions_explicit_config():
    """Test explicit EMBEDDINGS_DIMENSIONS configuration"""
    os.environ['EMBEDDINGS_DIMENSIONS'] = '1024'
    assert get_embedding_dimensions() == 1024

def test_get_embedding_dimensions_model_detection():
    """Test auto-detection based on model name"""
    os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
    os.environ.pop('EMBEDDINGS_DIMENSIONS', None)
    assert get_embedding_dimensions() == 1024

def test_collection_dimension_validation():
    """Test collection dimension validation logic"""
    client = QdrantClientWrapper()
    # Mock existing collection with different dimensions
    with patch.object(client.client, 'get_collection') as mock_get:
        mock_get.return_value = Mock(config=Mock(params=Mock(vectors=Mock(size=1536))))
        validation = client._validate_collection_dimensions('test', VectorParams(size=1024))
        assert validation['needs_recreation'] is True

def test_dimension_mismatch_recreation():
    """Test automatic collection recreation on dimension mismatch"""
    # Test the complete flow from detection to recreation
    pass

def test_deepinfra_api_integration():
    """Test DeepInfra API calls with correct model and dimensions"""
    # Mock DeepInfra API responses and test embedding creation
    pass
```

```bash
# Run iteratively until all tests pass:
uv run pytest tests/test_utils_integration.py -v
uv run pytest tests/test_qdrant_wrapper.py -v  
uv run pytest tests/test_deepinfra_integration.py -v

# If failing: Read error messages, understand root cause, fix code, re-run
# Never mock failures to pass - fix the underlying code
```

### Level 3: Integration Tests

```bash
# Test with actual DeepInfra configuration
export EMBEDDINGS_MODEL="Qwen/Qwen3-Embedding-0.6B"
export EMBEDDINGS_API_KEY="your_deepinfra_key"
export EMBEDDINGS_API_BASE="https://api.deepinfra.com/v1/openai"
export EMBEDDINGS_DIMENSIONS="1024"

# Start the MCP server
uv run python -m src.crawl4ai_mcp

# Expected startup logs:
# "Embedding setup validated - dimensions: 1024"
# "Created collection: crawled_pages" (if new) 
# "Collection crawled_pages has incompatible dimensions, recreating..." (if migrating)

# Test embedding creation
curl -X POST http://localhost:8000/test-endpoint \
  -H "Content-Type: application/json" \
  -d '{"text": "Test embedding creation"}'

# Expected: Successful embedding with 1024 dimensions
# If error: Check logs for API errors, dimension mismatches, or configuration issues
```

### Level 4: Migration & Creative Validation

```bash
# Test migration from 1536 to 1024 dimensions
echo "EMBEDDINGS_DIMENSIONS=1536" > .env.test
uv run python -c "from src.qdrant_wrapper import get_qdrant_client; get_qdrant_client()"

echo "EMBEDDINGS_DIMENSIONS=1024" > .env.test  
uv run python -c "from src.qdrant_wrapper import get_qdrant_client; get_qdrant_client()"

# Expected: Collections recreated with new dimensions

# Test OpenAI backward compatibility
export EMBEDDINGS_MODEL="text-embedding-3-small"
unset EMBEDDINGS_DIMENSIONS  # Should default to 1536
uv run python -c "from src.utils import get_embedding_dimensions; print(get_embedding_dimensions())"

# Expected: 1536 (auto-detected from model)

# Performance test dimension validation
time uv run python -c "from src.qdrant_wrapper import get_qdrant_client; get_qdrant_client()"

# Expected: <1 second startup time even with validation
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] DeepInfra integration works: Manual test with API key
- [ ] Dimension migration works: Test 1536→1024 migration
- [ ] Backward compatibility: OpenAI configs still work
- [ ] Error handling: Invalid configurations fail gracefully
- [ ] Logging: Informative messages for dimension changes
- [ ] Performance: Startup time impact minimal (<1s)

---

## Anti-Patterns to Avoid

- ❌ Don't hardcode any dimension values - use dynamic detection
- ❌ Don't skip collection validation - always check before creating
- ❌ Don't ignore dimension mismatches - handle them explicitly  
- ❌ Don't break backward compatibility - support existing configurations
- ❌ Don't bypass error handling - validate inputs and handle failures
- ❌ Don't remove existing test coverage - update and extend tests
- ❌ Don't ignore performance - validate startup time impact
- ❌ Don't create new environment variable patterns - follow existing conventions

## Confidence Level: 9/10

This PRP provides comprehensive context for one-pass implementation success through:
- Complete codebase analysis with specific file locations and line numbers
- Detailed external API research with documentation URLs and limitations
- Existing pattern analysis for environment variables, testing, and Qdrant integration
- Step-by-step implementation sequence with preserved compatibility
- Comprehensive validation approach following existing project patterns
- Risk mitigation for known gotchas and library quirks

The high confidence comes from thorough research into both the current implementation and target integration, providing the AI agent with all necessary context to implement successfully without additional research or guesswork.