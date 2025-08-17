# PRP Specification: MCP Tools Timeout Configuration

## Executive Summary

**Problem**: MCP client disconnections during large page processing due to default FastMCP timeouts
**Solution**: Implement granular timeout configuration for all MCP tools based on operation complexity
**Impact**: Eliminates timeouts, improves reliability, enforces timeout standards

---

## Current State Analysis

### 1. Current Implementation

```yaml
current_state:
  files: 
    - src/core/app.py:537-652 (register_tools function)
    - .env.example (environment configuration)
  
  behavior: |
    - All MCP tools use FastMCP default timeout
    - No timeout configuration per tool type
    - Uniform timeout regardless of operation complexity
    - Client disconnects on large operations (smart_crawl_url, index_github_repository)
  
  issues:
    - Client timeouts during large repository indexing
    - Web crawling failures on complex sites
    - No differentiation between quick vs long operations
    - Poor user experience with unexpected disconnections
```

### 2. Affected Tools Analysis

| Tool Category | Current State | Operation Type | Expected Duration |
|---------------|---------------|----------------|-------------------|
| **Web Tools** | No timeout | Long operations | 5-60 minutes |
| `crawl_single_page` | Default | Medium-Long | 1-30 minutes |
| `smart_crawl_url` | Default | Very Long | 10-60 minutes |
| **RAG Tools** | No timeout | Quick-Medium | 1-5 minutes |
| `get_available_sources` | Default | Quick | 10-60 seconds |
| `perform_rag_query` | Default | Medium | 1-5 minutes |
| `search_code_examples` | Default | Medium | 1-5 minutes |
| **GitHub Tools** | No timeout | Very Long | 15-60 minutes |
| `index_github_repository` | Default | Very Long | 30-60 minutes |
| **Knowledge Graph Tools** | No timeout | Medium-Long | 1-30 minutes |
| `check_ai_script_hallucinations` | Default | Medium | 1-10 minutes |
| `query_knowledge_graph` | Default | Quick | 10-60 seconds |

---

## Desired State Specification

### 1. Target Architecture

```yaml
desired_state:
  files:
    - src/core/app.py: Enhanced register_tools with timeout configuration
    - .env.example: New timeout environment variables
    - src/core/config.py: Centralized timeout configuration (optional)
  
  behavior: |
    - Granular timeout configuration per tool category
    - Environment-configurable timeout values
    - Appropriate timeouts for operation complexity
    - Zero timeout-related disconnections
    - Mandatory timeout enforcement
  
  benefits:
    - Eliminates client disconnections
    - Improved user experience
    - Configurable performance tuning
    - Production-ready reliability
    - Clear operation expectations
    - Enforced timeout standards
```

### 2. Timeout Strategy

```python
# Timeout Categories (seconds)
QUICK_TIMEOUT = 60        # 1 minute  - Simple queries, source listing
MEDIUM_TIMEOUT = 300      # 5 minutes - RAG queries, script analysis
LONG_TIMEOUT = 1800       # 30 minutes - Single page crawl, repository parsing
VERY_LONG_TIMEOUT = 3600  # 1 hour    - Multi-page crawl, full repository indexing
```

---

## Implementation Specification

### Phase 1: Core Timeout Configuration

#### Task 1.1: Environment Variables Configuration
```yaml
task_name: "Add timeout environment variables"
action: MODIFY
file: .env.example
changes: |
  - Add MCP Tools timeout section after line 224
  - Include 4 timeout categories with descriptions
  - Provide sensible defaults
  - Add usage guidelines
validation:
  - command: "grep 'MCP_.*_TIMEOUT' .env.example"
  - expect: "4 timeout variables found"
```

#### Task 1.2: Timeout Constants Definition
```yaml
task_name: "Define timeout constants in app.py"
action: MODIFY
file: src/core/app.py
changes: |
  - Add timeout configuration section after imports (line 10)
  - Import os for environment variable access
  - Define 4 timeout constants with environment fallbacks
  - Add documentation comments
validation:
  - command: "python -c 'from src.core.app import QUICK_TIMEOUT, MEDIUM_TIMEOUT; print(QUICK_TIMEOUT, MEDIUM_TIMEOUT)'"
  - expect: "Timeout values printed without errors"
```

### Phase 2: Tool Registration Enhancement

#### Task 2.1: Web Tools Timeout Application
```yaml
task_name: "Apply timeouts to web tools"
action: MODIFY
file: src/core/app.py
lines: 560-561
changes: |
  - Replace: app.tool()(web_tools.crawl_single_page)
  - With: app.tool(timeout=LONG_TIMEOUT)(web_tools.crawl_single_page)
  - Replace: app.tool()(web_tools.smart_crawl_url)
  - With: app.tool(timeout=VERY_LONG_TIMEOUT)(web_tools.smart_crawl_url)
validation:
  - command: "grep -n 'timeout=' src/core/app.py | grep 'web_tools'"
  - expect: "2 web tools with timeout configuration"
```

#### Task 2.2: RAG Tools Timeout Application
```yaml
task_name: "Apply timeouts to RAG tools"
action: MODIFY
file: src/core/app.py
lines: 617-618
changes: |
  - Replace: app.tool()(rag_tools.get_available_sources)
  - With: app.tool(timeout=QUICK_TIMEOUT)(rag_tools.get_available_sources)
  - Replace: app.tool()(rag_tools.perform_rag_query)
  - With: app.tool(timeout=MEDIUM_TIMEOUT)(rag_tools.perform_rag_query)
  - Update search_code_examples with MEDIUM_TIMEOUT
validation:
  - command: "grep -n 'timeout=' src/core/app.py | grep 'rag_tools'"
  - expect: "3 RAG tools with timeout configuration"
```

#### Task 2.3: GitHub Tools Timeout Application
```yaml
task_name: "Apply timeout to GitHub tools"
action: MODIFY
file: src/core/app.py
lines: 573-590
changes: |
  - Modify app.tool() call for index_github_repository
  - Add timeout=VERY_LONG_TIMEOUT parameter
  - Maintain existing name and description parameters
  - Preserve multi-line formatting
validation:
  - command: "grep -A 5 'index_github_repository' src/core/app.py | grep 'timeout='"
  - expect: "GitHub tool with very long timeout"
```

#### Task 2.4: Knowledge Graph Tools Timeout Application
```yaml
task_name: "Apply timeouts to knowledge graph tools"
action: MODIFY
file: src/core/app.py
lines: 635-637
changes: |
  - Replace: app.tool()(kg_tools.check_ai_script_hallucinations)
  - With: app.tool(timeout=MEDIUM_TIMEOUT)(kg_tools.check_ai_script_hallucinations)
  - Replace: app.tool()(kg_tools.query_knowledge_graph)
  - With: app.tool(timeout=QUICK_TIMEOUT)(kg_tools.query_knowledge_graph)
validation:
  - command: "grep -n 'timeout=' src/core/app.py | grep 'kg_tools'"
  - expect: "2 knowledge graph tools with timeout configuration"
```

### Phase 3: Validation and Testing

#### Task 3.1: Configuration Testing
```yaml
task_name: "Test timeout configuration loading"
action: CREATE
file: tests/unit/core/test_timeout_config.py
changes: |
  - Create unit test for timeout constant loading
  - Test environment variable fallbacks
  - Validate timeout value ranges
  - Test error handling for invalid values
validation:
  - command: "uv run pytest tests/unit/core/test_timeout_config.py -v"
  - expect: "All timeout configuration tests pass"
```

#### Task 3.2: Tool Registration Testing
```yaml
task_name: "Test tool registration with timeouts"
action: MODIFY
file: tests/unit/core/test_app.py
changes: |
  - Add timeout verification to existing tool registration tests
  - Test that all tools have appropriate timeout values
  - Verify timeout inheritance in tool instances
validation:
  - command: "uv run pytest tests/unit/core/test_app.py::test_register_tools -v"
  - expect: "Tool registration with timeouts verified"
```

#### Task 3.3: Integration Testing
```yaml
task_name: "Integration test for timeout behavior"
action: CREATE
file: tests/integration/test_timeout_behavior.py
changes: |
  - Create mock long-running operations
  - Test timeout enforcement
  - Verify graceful timeout handling
  - Test environment variable override
validation:
  - command: "uv run pytest tests/integration/test_timeout_behavior.py -v"
  - expect: "Timeout behavior correctly implemented"
```

### Phase 4: Documentation and Deployment

#### Task 4.1: Update Documentation
```yaml
task_name: "Update CLAUDE.md with timeout information"
action: MODIFY
file: CLAUDE.md
changes: |
  - Add timeout configuration section to Environment Configuration
  - Document timeout categories and their purposes
  - Add troubleshooting section for timeout issues
  - Include performance tuning guidelines
validation:
  - command: "grep -n 'TIMEOUT' CLAUDE.md"
  - expect: "Timeout documentation added"
```

#### Task 4.2: Deployment Verification
```yaml
task_name: "Test server startup with new configuration"
action: VALIDATE
changes: |
  - Start MCP server with new timeout configuration
  - Verify all tools register successfully
  - Test tool execution with timeout enforcement
  - Validate environment variable loading
validation:
  - command: "uv run -m src --test-mode"
  - expect: "Server starts successfully with timeout config"
```

---

## Technical Implementation Details

### 1. Environment Variables Schema

```bash
# MCP Tools Timeout Configuration (seconds)
# Controls maximum execution time for MCP tools to prevent client disconnections

# Quick operations: Simple queries, data retrieval
MCP_QUICK_TIMEOUT=60

# Medium operations: RAG queries, analysis tasks  
MCP_MEDIUM_TIMEOUT=300

# Long operations: Single page crawls, repository parsing
MCP_LONG_TIMEOUT=1800

# Very long operations: Multi-page crawls, full repository indexing
MCP_VERY_LONG_TIMEOUT=3600
```

### 2. Code Changes Summary

```python
# src/core/app.py - New timeout configuration section
import os
from typing import Final

# Timeout configuration (seconds)
QUICK_TIMEOUT: Final[int] = int(os.getenv("MCP_QUICK_TIMEOUT", "60"))
MEDIUM_TIMEOUT: Final[int] = int(os.getenv("MCP_MEDIUM_TIMEOUT", "300"))
LONG_TIMEOUT: Final[int] = int(os.getenv("MCP_LONG_TIMEOUT", "1800"))
VERY_LONG_TIMEOUT: Final[int] = int(os.getenv("MCP_VERY_LONG_TIMEOUT", "3600"))
```

### 3. Timeout Mapping Strategy

| Operation Type | Timeout | Rationale |
|----------------|---------|-----------|
| Data retrieval | QUICK | Simple database queries |
| Text processing | MEDIUM | RAG, analysis operations |
| Web crawling | LONG | Network-dependent operations |
| Repository processing | VERY_LONG | Large data processing |

---

## Risk Assessment and Mitigation

### High-Risk Areas

1. **Breaking Changes Risk: MINIMAL**
   - Mitigation: Explicit timeout enforcement improves reliability
   - Impact: Tools will now timeout appropriately instead of hanging

2. **Performance Impact Risk: VERY LOW**
   - Mitigation: Constants evaluated once at startup
   - Monitoring: No runtime performance overhead

3. **Configuration Complexity Risk: LOW**
   - Mitigation: Sensible defaults, clear documentation
   - Impact: Required timeout values force proper configuration

### Rollback Strategy

1. **Immediate**: Remove timeout parameters from tool registrations
2. **Environment**: Remove new environment variables
3. **Constants**: Remove timeout constant definitions
4. **Tests**: Disable timeout-related tests

---

## Success Metrics

### Primary Objectives
- [ ] Zero timeout-related MCP client disconnections
- [ ] All 9 MCP tools have mandatory timeout configuration
- [ ] Environment variables working correctly
- [ ] No default timeout fallbacks (explicit configuration required)

### Performance Metrics
- [ ] Server startup time unchanged
- [ ] Tool registration time < 100ms additional overhead
- [ ] Configuration loading < 10ms

### Quality Metrics
- [ ] 100% test coverage for timeout functionality
- [ ] All tools enforce explicit timeout values
- [ ] Documentation updated and accurate
- [ ] No dead code or unused fallback paths

---

## Dependencies and Prerequisites

### Technical Dependencies
- FastMCP framework timeout parameter support ✓
- Python environment variable access ✓
- Existing tool registration architecture ✓

### Testing Dependencies
- pytest framework ✓
- Mock objects for timeout testing ✓
- Integration test infrastructure ✓

### Deployment Dependencies
- Environment variable configuration capability ✓
- Docker Compose service restart capability ✓
- No breaking changes to MCP protocol ✓

---

## Implementation Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|-------------|
| **Phase 1** | 2 hours | None | Environment config, constants |
| **Phase 2** | 3 hours | Phase 1 | Tool timeout application |
| **Phase 3** | 4 hours | Phase 2 | Testing and validation |
| **Phase 4** | 1 hour | Phase 3 | Documentation, deployment |

**Total Estimated Time: 10 hours**

---

## Validation Checklist

### Pre-Implementation
- [ ] Current timeout behavior documented
- [ ] Tool registration points identified
- [ ] Environment variable strategy defined

### During Implementation
- [ ] Each tool timeout configured appropriately
- [ ] Environment variables added with defaults
- [ ] Tests written and passing
- [ ] No breaking changes introduced

### Post-Implementation
- [ ] Large repository indexing works without timeout
- [ ] Smart crawl operations complete successfully
- [ ] All tools have explicit timeout enforcement
- [ ] Performance benchmarks maintained
- [ ] No legacy timeout handling code remains

---

## Future Enhancements

### Phase 2 Considerations
- **Dynamic Timeout Adjustment**: Based on operation complexity detection
- **Timeout Monitoring**: Metrics collection for timeout optimization
- **User Feedback**: Progress indicators for long-running operations
- **Adaptive Timeouts**: Machine learning-based timeout prediction

### Integration Opportunities
- **Health Checks**: Timeout-aware service health monitoring
- **Load Balancing**: Timeout consideration in request routing
- **Caching**: Timeout-based cache invalidation strategies

---

*This specification provides a comprehensive roadmap for implementing granular timeout configuration across all MCP tools, ensuring reliable operation for large-scale processing tasks with explicit timeout enforcement and clean code architecture.*