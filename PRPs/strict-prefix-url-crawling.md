# PRP: Strict Prefix URL Crawling Implementation

## Executive Summary

**Transformation Goal**: Implement strict prefix URL filtering in the `smart_crawl_url` tool to ensure crawling remains within the specified URL scope, preventing indexation of irrelevant content such as different language sections or unrelated domains.

**Business Impact**: Dramatically improve content relevance and precision while reducing resource waste and improving user experience by ensuring focused crawling scope.

---

## Current State Analysis

### Current Implementation
```yaml
current_state:
  files:
    - src/tools/web_tools.py (crawl_recursive_internal_links)
    - src/tools/web_tools.py (smart_crawl_url)
    - tests/test_web_tools.py (existing tests)
  behavior:
    - crawl_recursive_internal_links follows ALL internal links discovered
    - No filtering mechanism for URL prefixes
    - Can crawl across different language sections (e.g., /en to /pt)
    - No scope restrictions beyond internal vs external link detection
  issues:
    - Content pollution: Mixes different language content
    - Resource waste: Crawls irrelevant sections
    - Poor precision: Indexation of out-of-scope content
    - User confusion: Results contain mixed contexts
```

### Technical Debt Identified
- **Missing URL Scope Control**: No mechanism to restrict crawling to specific URL patterns
- **Imprecise Link Filtering**: Current filtering only distinguishes internal vs external links
- **Test Coverage Gap**: No tests for URL filtering behaviors
- **Performance Inefficiency**: Wastes resources on irrelevant content

---

## Desired State Architecture

### Target Implementation
```yaml
desired_state:
  files:
    - src/tools/web_tools.py (enhanced crawl_recursive_internal_links with base_prefix)
    - src/tools/web_tools.py (updated smart_crawl_url integration)
    - tests/test_web_tools.py (new prefix filtering tests)
  behavior:
    - crawl_recursive_internal_links respects base_prefix parameter
    - Smart filtering: Only URLs starting with base_prefix are crawled
    - Language-specific scope: /en URLs only crawl other /en URLs
    - Maintains existing functionality for sitemaps and text files
  benefits:
    - 90%+ improvement in content relevance
    - 40-60% reduction in crawling time for scoped operations
    - Better user experience with focused results
    - Resource optimization and cost reduction
```

---

## Hierarchical Objectives

### 1. High-Level Objective
**Transform crawling behavior to respect strict URL prefix boundaries**

### 2. Mid-Level Milestones

#### Milestone 1: Core Function Enhancement (Priority: HIGH)
- Add base_prefix parameter to crawl_recursive_internal_links
- Implement URL filtering logic with normalization
- Ensure backward compatibility

#### Milestone 2: Integration Update (Priority: HIGH)  
- Update smart_crawl_url to pass base_prefix
- Verify existing functionality is preserved
- Maintain current API contracts

#### Milestone 3: Testing & Validation (Priority: CRITICAL)
- Create comprehensive test suite for new behavior
- Ensure all existing tests continue to pass
- Add edge case coverage

### 3. Low-Level Tasks

#### Task 1.1: Function Signature Enhancement
```yaml
task_name: enhance_crawl_recursive_internal_links_signature
action: MODIFY
file: src/tools/web_tools.py
changes: |
  - Add base_prefix: str parameter to function signature
  - Update docstring with parameter documentation
  - Add type hints and parameter validation
validation:
  - command: "uv run python -c 'from src.tools.web_tools import crawl_recursive_internal_links; import inspect; print(inspect.signature(crawl_recursive_internal_links))'"
  - expect: "base_prefix parameter present in signature"
```

#### Task 1.2: URL Filtering Logic Implementation
```yaml
task_name: implement_prefix_filtering_logic
action: MODIFY
file: src/tools/web_tools.py
changes: |
  - Modify the internal link processing loop
  - Add normalize_url(link["href"]).startswith(base_prefix) condition
  - Preserve existing normalize_url functionality
  - Add logging for filtered URLs (debug level)
validation:
  - command: "uv run pytest tests/test_web_tools.py::TestWebToolsCrawlingFunctions::test_crawl_recursive_internal_links -v"
  - expect: "test passes with new parameter"
```

#### Task 2.1: Smart Crawl URL Integration
```yaml
task_name: update_smart_crawl_url_integration
action: MODIFY
file: src/tools/web_tools.py
changes: |
  - Update crawl_recursive_internal_links call in smart_crawl_url
  - Pass base_prefix=url parameter
  - Ensure regular webpage crawling uses the filter
  - Preserve sitemap and txt file behavior
validation:
  - command: "uv run pytest tests/test_web_tools.py::TestWebToolsCrawlingFunctions::test_smart_crawl_url_regular_webpage -v"
  - expect: "test passes with updated integration"
```

#### Task 3.1: Core Test Implementation
```yaml
task_name: create_prefix_filtering_tests
action: ADD
file: tests/test_web_tools.py
changes: |
  - Add test_crawl_recursive_internal_links_respects_base_prefix method
  - Mock crawler responses with mixed URL prefixes
  - Verify only matching prefix URLs are processed
  - Test edge cases: trailing slashes, query parameters, fragments
validation:
  - command: "uv run pytest tests/test_web_tools.py::TestWebToolsCrawlingFunctions::test_crawl_recursive_internal_links_respects_base_prefix -v"
  - expect: "new test passes"
```

#### Task 3.2: Integration Test Enhancement
```yaml
task_name: enhance_integration_tests
action: MODIFY
file: tests/test_web_tools.py
changes: |
  - Update existing tests to handle new parameter
  - Add specific test cases for language separation
  - Test backward compatibility scenarios
  - Add performance regression tests
validation:
  - command: "uv run pytest tests/test_web_tools.py -v"
  - expect: "all tests pass"
```

#### Task 3.3: End-to-End Validation
```yaml
task_name: comprehensive_test_validation
action: MODIFY
file: tests/test_web_tools.py
changes: |
  - Add realistic URL filtering scenarios
  - Test with actual URL patterns (docs.anthropic.com/en vs /pt)
  - Validate crawl result URL consistency
  - Performance benchmarking for filtering overhead
validation:
  - command: "uv run pytest tests/test_web_tools.py -v --tb=short"
  - expect: "all tests pass with improved coverage"
```

---

## Implementation Strategy

### Phase 1: Core Enhancement (Days 1-2)
1. **Dependency Analysis**: Review current crawl_recursive_internal_links usage
2. **Function Modification**: Implement base_prefix parameter and filtering logic
3. **Unit Testing**: Create isolated tests for the new functionality

### Phase 2: Integration (Days 2-3)
1. **Smart Crawl Update**: Integrate prefix filtering into smart_crawl_url
2. **Backward Compatibility**: Ensure existing behavior is preserved
3. **Integration Testing**: Validate end-to-end functionality

### Phase 3: Validation & Optimization (Days 3-4)
1. **Comprehensive Testing**: Full test suite execution
2. **Performance Validation**: Ensure no significant performance regression
3. **Documentation Update**: Update function docstrings and comments

### Rollback Plan
1. **Git Branch Strategy**: Implement in feature branch with clear commit history
2. **Automated Rollback**: If tests fail, revert specific commits
3. **Fallback Logic**: Add feature flag to enable/disable strict filtering

---

## Risk Assessment & Mitigation

### High-Priority Risks

#### Risk 1: Backward Compatibility Break
- **Probability**: Medium (30%)
- **Impact**: High
- **Mitigation**: Add default parameter value, extensive testing
- **Contingency**: Feature flag implementation

#### Risk 2: Performance Regression  
- **Probability**: Low (15%)
- **Impact**: Medium
- **Mitigation**: Efficient string operations, performance benchmarking
- **Contingency**: Optimize filtering algorithm

#### Risk 3: Edge Case URL Handling
- **Probability**: Medium (25%) 
- **Impact**: Medium
- **Mitigation**: Comprehensive URL normalization testing
- **Contingency**: Enhanced normalize_url function

### Medium-Priority Risks

#### Risk 4: Test Coverage Gaps
- **Probability**: Low (20%)
- **Impact**: Medium
- **Mitigation**: Systematic test case development
- **Contingency**: Post-deployment monitoring

---

## Validation Criteria

### Functional Validation
- [ ] `crawl_recursive_internal_links` accepts and uses base_prefix parameter
- [ ] URLs not matching base_prefix are filtered out during crawling
- [ ] Existing functionality for sitemaps and txt files remains unchanged
- [ ] All existing tests pass without modification
- [ ] New tests validate prefix filtering behavior

### Performance Validation
- [ ] No significant performance regression (< 5% slowdown)
- [ ] Actual improvement in crawling efficiency for scoped operations
- [ ] Memory usage remains stable

### Quality Validation
- [ ] Code follows project style guidelines
- [ ] Comprehensive docstring documentation
- [ ] Type hints are complete and accurate
- [ ] Error handling is robust

---

## Success Metrics

### Primary Metrics
- **Content Relevance**: 90%+ of crawled URLs match the specified prefix
- **Resource Efficiency**: 40-60% reduction in irrelevant page processing
- **Test Coverage**: 100% of new functionality covered by tests

### Secondary Metrics
- **Performance Impact**: < 5% overhead for filtering logic
- **Backward Compatibility**: 100% existing functionality preserved
- **Documentation Quality**: Complete type hints and docstring coverage

---

## Dependencies & Integration Points

### Internal Dependencies
- `crawl4ai` library for web crawling functionality
- `urllib.parse.urldefrag` for URL normalization
- Existing test framework and mocking patterns

### External Integration Points
- MCP tool registration and context handling
- Qdrant vector database integration (unchanged)
- Docker service dependencies (unchanged)

### Cross-System Impacts
- **No Breaking Changes**: Existing MCP tool contracts preserved
- **Enhanced Precision**: Improved RAG query results due to focused content
- **Resource Optimization**: Reduced database storage for irrelevant content

---

## Timeline & Milestones

### Week 1: Implementation
- **Days 1-2**: Core function enhancement and unit testing
- **Days 3-4**: Integration updates and comprehensive testing
- **Days 4-5**: Validation, optimization, and documentation

### Milestone Gates
1. **Gate 1** (Day 2): Core functionality implemented and unit tested
2. **Gate 2** (Day 4): Integration complete with all tests passing
3. **Gate 3** (Day 5): Performance validated and documentation complete

### Go/No-Go Criteria
- **Go Criteria**: All tests pass, performance acceptable, functionality validated
- **No-Go Criteria**: Test failures, significant performance regression, integration issues

---

## Post-Implementation Monitoring

### Health Checks
- Monitor crawling precision metrics
- Track resource usage patterns
- Validate user satisfaction with result relevance

### Performance Monitoring
- Crawling time measurements for different URL patterns
- Memory usage during large-scale crawling operations
- Error rate monitoring for edge cases

### Continuous Improvement
- Gather user feedback on crawling precision
- Identify additional filtering needs
- Consider performance optimization opportunities

---

## Technical Implementation Details

### Code Changes Summary

#### File: `src/tools/web_tools.py`
```python
# Modified function signature
async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    base_prefix: str,  # NEW PARAMETER
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:

# Modified link processing loop
for link in result.links.get("internal", []):
    next_url = normalize_url(link["href"])
    if next_url not in visited and next_url.startswith(base_prefix):  # NEW CONDITION
        next_level_urls.add(next_url)

# Updated smart_crawl_url call
crawl_results = await crawl_recursive_internal_links(
    crawler, [url], base_prefix=url, max_depth=max_depth, max_concurrent=max_concurrent
)
```

#### File: `tests/test_web_tools.py`
```python
@pytest.mark.asyncio
async def test_crawl_recursive_internal_links_respects_base_prefix(self):
    """Test that crawl_recursive_internal_links respects base_prefix filtering."""
    # Mock setup with mixed URL prefixes
    # Validation that only matching URLs are processed
    # Edge case testing
```

### Configuration Management
- No environment variable changes required
- No database schema modifications
- Backward compatible API design

---

## Approval Checklist

- [ ] **Technical Review**: Code changes reviewed by senior developer
- [ ] **Test Coverage**: Comprehensive test suite implemented and passing
- [ ] **Performance**: Benchmarks show acceptable performance characteristics
- [ ] **Documentation**: All code changes properly documented
- [ ] **Backward Compatibility**: Existing functionality verified intact
- [ ] **Integration**: MCP tool integration tested and validated
- [ ] **User Acceptance**: Meets all specified acceptance criteria

---

**PRP Document Version**: 1.0  
**Created**: 2025-01-08  
**Status**: Ready for Implementation  
**Priority**: High  
**Estimated Effort**: 4-5 days  
**Risk Level**: Medium-Low