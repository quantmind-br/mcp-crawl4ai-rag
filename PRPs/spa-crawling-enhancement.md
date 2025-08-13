# SPA Crawling Enhancement PRP

---

## Goal

**Feature Goal**: Enhance the `smart_crawl_url` tool to detect and crawl Single Page Applications (SPAs) with comprehensive framework support, increasing documentation site coverage from 1 to 29+ pages (2900% improvement) while maintaining strict URL scope filtering.

**Deliverable**: Enhanced web crawling system with SPA detection, framework-specific URL extraction, and JavaScript execution capabilities that seamlessly integrates with existing crawling infrastructure.

**Success Definition**: 
- Flow Launcher docs crawls 29 pages instead of 1 (target validation site)
- Zero breaking changes to existing crawling functionality
- Scope filtering maintains security (e.g., `/docs/` URLs only, no `/plugins/` leakage)
- Performance impact <10% for non-SPA sites
- Full test coverage with both unit and integration tests

## User Persona

**Target User**: AI developers and researchers using MCP tools for documentation crawling and knowledge base creation

**Use Case**: Crawling modern documentation sites built with SPA frameworks (Docsify, GitBook, VuePress, Docusaurus) for RAG applications and AI knowledge base construction

**User Journey**: 
1. User calls `smart_crawl_url` with SPA documentation URL (e.g., `https://www.flowlauncher.com/docs/`)
2. System detects SPA framework automatically
3. Tool extracts all navigation URLs while respecting scope
4. Content from all pages is crawled and stored in vector database
5. User can query comprehensive documentation content via RAG tools

**Pain Points Addressed**: 
- Current crawler only captures 1 page from SPA sites (96.5% content loss)
- Manual URL discovery required for comprehensive coverage
- JavaScript-heavy sites not properly crawled
- Documentation fragmentation due to incomplete indexing

## Why

- **Business Value**: Modern documentation increasingly uses SPA frameworks - without SPA support, RAG systems miss 95%+ of available content
- **Integration**: Seamlessly extends existing `smart_crawl_url` tool without disrupting current workflows
- **Performance**: SPAs often provide better structured navigation than traditional sites, enabling more efficient bulk crawling
- **Future-Proofing**: Positions the crawling system for modern web architecture trends

## What

Implement cascading SPA detection with framework-specific URL extraction while maintaining backward compatibility and security.

### Success Criteria

- [ ] **SPA Detection**: Automatically identify 4+ framework types (Docsify, GitBook, VuePress, Docusaurus)
- [ ] **URL Extraction**: Extract navigation URLs with 95%+ accuracy for supported frameworks  
- [ ] **Scope Preservation**: Maintain existing `base_prefix` filtering with zero security regressions
- [ ] **Performance**: Non-SPA sites experience <10% performance impact
- [ ] **Framework Coverage**: Support hash routing (`#/page`) and history API routing (`/page`)
- [ ] **Error Resilience**: Graceful fallback to existing crawler for unsupported/failed SPA detection
- [ ] **Test Coverage**: 95%+ code coverage with comprehensive edge case testing

## All Needed Context

### Context Completeness Check

_Validated: An AI agent with no prior knowledge of this codebase can implement this successfully using only this PRP content and codebase access._

### Documentation & References

```yaml
# MUST READ - Critical implementation references
- file: src/tools/web_tools.py:425-487
  why: crawl_recursive_internal_links function shows existing URL filtering pattern with base_prefix
  pattern: Scope restriction using next_url.startswith(base_prefix)
  gotcha: urldefrag() strips hash fragments - SPA URLs need special handling

- file: src/tools/web_tools.py:658-694  
  why: create_base_prefix function shows scope generation pattern to follow
  pattern: URL prefix normalization with trailing slash logic
  gotcha: Must preserve this exact logic for security scope enforcement

- file: src/tools/web_tools.py:697-930
  why: smart_crawl_url main function shows integration point for SPA detection
  pattern: Type detection then strategy selection (sitemap vs recursive vs new SPA)
  gotcha: Context access pattern and error handling structure to maintain

- url: https://docs.crawl4ai.com/advanced/session-management/#integrated-javascript-execution-and-waiting
  why: JavaScript execution patterns for dynamic content loading
  critical: wait_for parameter usage and networkidle detection for SPA content

- url: https://docs.crawl4ai.com/core/examples/
  why: CrawlerRunConfig examples with js_code and wait_for parameters
  critical: Proper async/await patterns and error handling for JS execution

- docfile: PRPs/ai_docs/spa_crawling_reference.md
  why: Comprehensive BeautifulSoup patterns and urllib.parse usage for SPA detection
  section: All sections - contains production-ready code patterns
```

### Current Codebase Tree (relevant sections)

```bash
src/
├── tools/
│   ├── web_tools.py                 # Main integration point - smart_crawl_url()
│   └── __init__.py
├── services/
│   ├── rag_service.py              # Storage integration patterns
│   └── __init__.py  
├── core/
│   ├── context.py                  # Context access patterns
│   └── app.py                      # MCP tool registration
tests/
├── unit/tools/
│   ├── test_web_tools.py           # Testing patterns to follow
│   └── __init__.py
├── conftest.py                     # Mock fixtures and context setup
└── fixtures/                      # Test data samples
```

### Desired Codebase Tree with Files Added

```bash
src/
├── tools/
│   ├── web_tools.py                 # MODIFY: Add SPA detection functions
│   ├── spa_detector.py              # CREATE: Framework detection logic
│   └── __init__.py
├── services/
│   ├── rag_service.py              # UNCHANGED: Storage integration
│   └── __init__.py
tests/
├── unit/tools/
│   ├── test_web_tools.py           # MODIFY: Add SPA crawling tests  
│   ├── test_spa_detector.py        # CREATE: SPA detection tests
│   └── __init__.py
├── integration/
│   ├── test_spa_integration.py     # CREATE: End-to-end SPA crawling tests
│   └── __init__.py
├── conftest.py                     # MODIFY: Add SPA mock fixtures
└── fixtures/
    ├── spa_docsify.html            # CREATE: Test data for Docsify detection
    ├── spa_gitbook.html            # CREATE: Test data for GitBook detection
    └── spa_vuepress.html           # CREATE: Test data for VuePress detection
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: Windows Unicode Compatibility
# All console output must use ASCII-only characters
print("SUCCESS: SPA detected")  # ✅ Correct
print("✅ SPA detected")        # ❌ Causes UnicodeEncodeError on Windows

# CRITICAL: URL Scope Security  
# SPA URL extraction MUST respect base_prefix filtering
normalized_url = urldefrag(full_url)[0]  # Remove hash for comparison
if normalized_url.startswith(base_prefix):  # Same logic as existing crawler
    urls.add(full_url)

# CRITICAL: Crawl4AI Context Access
# Always access crawler through context, not direct instantiation
crawler = ctx.request_context.lifespan_context.crawler  # ✅ Correct
qdrant_client = ctx.request_context.lifespan_context.qdrant_client

# CRITICAL: Storage Dependencies
# MUST call update_source_info BEFORE add_documents_to_vector_db
update_source_info(qdrant_client, source_id, summary, word_count)  # FIRST
add_documents_to_vector_db(qdrant_client, urls, chunks, metadatas)  # SECOND

# CRITICAL: Async/Await Error Handling
# Use proper exception handling for JavaScript execution failures
try:
    result = await crawler.arun(url=url, config=js_config)
    if not result.success:
        # Fallback to standard crawling - do NOT raise exception
        return await standard_crawl_fallback(...)
except Exception:
    # Graceful degradation on JS failures
    return await standard_crawl_fallback(...)
```

## Implementation Blueprint

### Data Models and Structure

The SPA enhancement requires new configuration and detection models:

```python
# File: src/tools/spa_detector.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class SPAFramework(Enum):
    DOCSIFY = "docsify"
    GITBOOK = "gitbook" 
    VUEPRESS = "vuepress"
    DOCUSAURUS = "docusaurus"
    UNKNOWN = "unknown"

@dataclass
class SPADetectionResult:
    """Result of SPA framework detection."""
    is_spa: bool
    framework: SPAFramework
    confidence: float
    navigation_urls: List[str]
    requires_javascript: bool
    detection_patterns: List[str]

@dataclass  
class SPAConfig:
    """Configuration for SPA crawling."""
    enable_javascript: bool = True
    wait_timeout: int = 10000  # milliseconds
    max_wait_time: int = 15000
    concurrent_limit: int = 3
    fallback_enabled: bool = True
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/tools/spa_detector.py
  - IMPLEMENT: SPAFramework enum, SPADetectionResult, SPAConfig dataclasses
  - IMPLEMENT: detect_spa_framework(html_content: str, url: str) -> SPADetectionResult
  - IMPLEMENT: extract_navigation_urls(html_content: str, framework: SPAFramework, base_url: str) -> List[str]
  - FOLLOW pattern: src/tools/web_tools.py (function-based structure, proper error handling)
  - NAMING: snake_case functions, CamelCase classes, descriptive variable names
  - PLACEMENT: New module in src/tools/ for framework detection logic

Task 2: MODIFY src/tools/web_tools.py - Add SPA detection integration
  - IMPLEMENT: Integrate detect_spa_framework() into smart_crawl_url() after line 730
  - IMPLEMENT: extract_docsify_urls(), extract_gitbook_urls() functions with scope filtering
  - FOLLOW pattern: Existing type detection logic (sitemap, txt, now spa)
  - PRESERVE: Existing base_prefix generation and scope filtering logic
  - NAMING: extract_{framework}_urls naming convention
  - DEPENDENCIES: Import spa_detector functions

Task 3: CREATE tests/unit/tools/test_spa_detector.py
  - IMPLEMENT: Unit tests for framework detection (all 4 frameworks + negative cases)
  - IMPLEMENT: URL extraction tests with scope filtering validation
  - FOLLOW pattern: tests/unit/tools/test_web_tools.py (mock structure, assertion patterns)
  - NAMING: test_{function}_{scenario} naming convention
  - COVERAGE: All detection patterns, edge cases, malformed HTML handling
  - PLACEMENT: Unit tests in tests/unit/tools/

Task 4: MODIFY tests/unit/tools/test_web_tools.py
  - IMPLEMENT: SPA crawling integration tests with mock crawler
  - IMPLEMENT: Scope filtering security tests (verify no URL leakage)
  - FOLLOW pattern: Existing test structure with mock context and services
  - MOCK: AsyncWebCrawler with SPA-specific results, service layer functions
  - COVERAGE: SPA detection flow, JavaScript execution, error fallback
  - PRESERVE: All existing test cases

Task 5: CREATE tests/fixtures/spa_test_data.py
  - IMPLEMENT: Mock HTML content for each framework (realistic navigation structures)
  - IMPLEMENT: Mock crawl results with dynamic content simulation
  - FOLLOW pattern: tests/conftest.py fixture structure
  - NAMING: spa_{framework}_html fixtures, descriptive test data
  - COVERAGE: Valid framework examples, malformed HTML, mixed content
  - PLACEMENT: Test fixtures in tests/fixtures/

Task 6: CREATE tests/integration/test_spa_integration.py
  - IMPLEMENT: End-to-end SPA crawling workflow tests
  - IMPLEMENT: Performance benchmark tests (<10% overhead validation)
  - FOLLOW pattern: tests/integration/ structure with async test patterns
  - MOCK: Full context with qdrant_client and crawler
  - COVERAGE: Complete SPA crawling flow, storage integration, error scenarios
  - PLACEMENT: Integration tests in tests/integration/
```

### Implementation Patterns & Key Details

```python
# Framework Detection Pattern (src/tools/spa_detector.py)
def detect_spa_framework(html_content: str, url: str) -> SPADetectionResult:
    """
    Detect SPA framework using multiple detection methods.
    PATTERN: Cascading detection with confidence scoring
    """
    # Method 1: Meta tag detection
    meta_generators = extract_meta_generators(html_content)
    
    # Method 2: Script tag patterns  
    script_patterns = find_framework_scripts(html_content)
    
    # Method 3: DOM structure patterns
    dom_patterns = analyze_dom_structure(html_content)
    
    # CRITICAL: Confidence-based selection with fallback
    framework, confidence = calculate_framework_confidence(
        meta_generators, script_patterns, dom_patterns
    )
    
    return SPADetectionResult(
        is_spa=confidence > 0.6,  # Threshold based on research
        framework=framework,
        confidence=confidence,
        navigation_urls=[],  # Filled by extract_navigation_urls
        requires_javascript=framework in [SPAFramework.DOCSIFY, SPAFramework.VUEPRESS]
    )

# URL Extraction with Scope Security (src/tools/web_tools.py)
def extract_docsify_urls(base_url: str, html_content: str, base_prefix: str) -> List[str]:
    """
    Extract Docsify navigation URLs with mandatory scope filtering.
    CRITICAL: Must preserve existing security patterns
    """
    from bs4 import BeautifulSoup
    from urllib.parse import urldefrag, urljoin, urlparse
    
    soup = BeautifulSoup(html_content, 'html.parser')
    urls = set()
    
    # Extract from sidebar navigation (primary method)
    nav_links = soup.select('aside.sidebar a[href^="#/"], nav.app-nav a[href^="#/"]')
    for link in nav_links:
        href = link.get('href')
        if href and href.startswith('#/'):
            full_url = f"{base_url.rstrip('/')}{href}"
            
            # CRITICAL: Apply same scope filtering as existing crawler
            normalized_url = urldefrag(full_url)[0]  # Remove hash for comparison
            if normalized_url.startswith(base_prefix):
                urls.add(full_url)
    
    # SECURITY: Additional domain validation
    base_domain = urlparse(base_url).netloc
    filtered_urls = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc == base_domain:  # Same domain only
            filtered_urls.append(url)
    
    return filtered_urls

# Integration Pattern (src/tools/web_tools.py)
# INSERT after line 730 in smart_crawl_url function:
else:
    # NEW: SPA detection before recursive crawl
    initial_result = await crawler.arun(url=url)
    if initial_result.success and initial_result.html:
        from .spa_detector import detect_spa_framework
        
        detection = detect_spa_framework(initial_result.html, url)
        
        if detection.is_spa and detection.framework == SPAFramework.DOCSIFY:
            # Extract URLs with scope filtering
            base_prefix = create_base_prefix(url)  # Use existing security function
            spa_urls = extract_docsify_urls(url, initial_result.html, base_prefix)
            
            if spa_urls:
                # Use existing batch crawler for consistency
                crawl_results = await crawl_batch(
                    crawler, spa_urls, max_concurrent=max_concurrent
                )
                crawl_type = "spa_docsify"
            else:
                # Fallback with original result
                crawl_results = [{"url": url, "markdown": initial_result.markdown}]
                crawl_type = "webpage"
        else:
            # Standard recursive crawl for non-SPA or unsupported frameworks
            base_prefix = create_base_prefix(url)
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], base_prefix=base_prefix,
                max_depth=max_depth, max_concurrent=max_concurrent
            )
            crawl_type = "webpage"
```

### Integration Points

```yaml
CONTEXT_ACCESS:
  - pattern: "ctx.request_context.lifespan_context.crawler"
  - why: "Consistent with existing tool patterns for async crawler access"

SERVICE_INTEGRATION:
  - imports: "from ..services.rag_service import add_documents_to_vector_db, update_source_info"
  - pattern: "Function imports, not class instantiation for service layer"

ERROR_HANDLING:
  - pattern: "Graceful fallback to existing crawler on SPA detection/extraction failures"
  - why: "Zero breaking changes requirement - SPA features must be additive only"

CONFIGURATION:
  - add_to: "Environment variables in .env.example"
  - pattern: "ENABLE_SPA_CRAWLING=true, SPA_TIMEOUT_MS=10000"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
ruff check src/tools/spa_detector.py --fix
ruff check src/tools/web_tools.py --fix  
mypy src/tools/spa_detector.py
mypy src/tools/web_tools.py

# Project-wide validation
ruff check src/ --fix
mypy src/
ruff format src/

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test new SPA detection module
uv run pytest tests/unit/tools/test_spa_detector.py -v

# Test web tools integration 
uv run pytest tests/unit/tools/test_web_tools.py::TestSPACrawling -v

# Full test suite for tools
uv run pytest tests/unit/tools/ -v

# Coverage validation
uv run pytest tests/unit/tools/ --cov=src.tools --cov-report=term-missing

# Expected: All tests pass, coverage >95% for new code
```

### Level 3: Integration Testing (System Validation)

```bash
# End-to-end SPA crawling workflow
uv run pytest tests/integration/test_spa_integration.py -v

# Validate with actual Flow Launcher docs (if accessible)
uv run python -c "
import asyncio
import json
from src.tools.web_tools import smart_crawl_url
from src.core.context import create_test_context

async def test_flow_launcher():
    ctx = create_test_context()
    result = await smart_crawl_url(
        ctx, 'https://www.flowlauncher.com/docs/',
        max_depth=3, max_concurrent=5
    )
    data = json.loads(result)
    print(f'Success: {data[\"success\"]}')
    print(f'Pages crawled: {data[\"pages_crawled\"]}')
    print(f'Crawl type: {data[\"crawl_type\"]}')
    
    # Validation: Should crawl 25+ pages vs 1 for baseline
    assert data['success'] == True
    assert data['pages_crawled'] >= 25
    assert 'spa' in data.get('crawl_type', '')

asyncio.run(test_flow_launcher())
"

# Expected: 25+ pages crawled vs 1 baseline, crawl_type contains 'spa'
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Performance Impact Testing
uv run python -c "
import asyncio
import time
import json
from src.tools.web_tools import smart_crawl_url

async def performance_test():
    # Test non-SPA site performance impact
    start = time.time()
    # Use known non-SPA site for baseline
    result = await smart_crawl_url(ctx, 'https://example.com/', max_depth=2)
    baseline_time = time.time() - start
    
    print(f'Non-SPA crawl time: {baseline_time:.2f}s')
    # Expected: <10% performance impact vs previous implementation
    
asyncio.run(performance_test())
"

# Security Scope Testing  
uv run python -c "
import asyncio
import json
from src.tools.web_tools import smart_crawl_url

async def security_test():
    # Test scope filtering with docs URL
    result = await smart_crawl_url(
        ctx, 'https://www.flowlauncher.com/docs/', max_depth=3
    )
    data = json.loads(result)
    
    # Verify no URLs outside /docs/ scope
    crawled_urls = data.get('urls_crawled', [])
    for url in crawled_urls:
        assert '/docs/' in url, f'URL outside scope: {url}'
        assert '/plugins/' not in url, f'Security leak: {url}'
    
    print('SECURITY: All URLs within scope')

asyncio.run(security_test())
"

# Framework Detection Accuracy
uv run python -c "
from src.tools.spa_detector import detect_spa_framework

# Test with known framework samples
docsify_html = open('tests/fixtures/spa_docsify.html').read()
result = detect_spa_framework(docsify_html, 'https://test.com/docs/')

assert result.framework == SPAFramework.DOCSIFY
assert result.confidence > 0.8
print('DETECTION: Framework identification successful')
"

# Expected: All validations pass, performance <10% impact, security maintained
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest tests/unit/tools/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] No formatting issues: `uv run ruff format src/ --check`

### Feature Validation

- [ ] Flow Launcher docs crawls 25+ pages vs 1 baseline
- [ ] SPA detection works for all 4 supported frameworks (Docsify, GitBook, VuePress, Docusaurus)
- [ ] Scope filtering prevents URL leakage (no `/plugins/` URLs when crawling `/docs/`)
- [ ] Performance impact <10% for non-SPA sites
- [ ] Graceful fallback to existing crawler for unsupported sites

### Code Quality Validation

- [ ] Follows existing codebase patterns (function-based tools, context access)
- [ ] File placement matches desired codebase tree structure
- [ ] Anti-patterns avoided (no new dependencies, no breaking changes)
- [ ] Integration points work as specified (service layer, storage)
- [ ] Windows Unicode compatibility (no emoji characters in output)

### Security & Deployment

- [ ] URL scope filtering security maintained (same `base_prefix` logic)
- [ ] No new security vulnerabilities introduced
- [ ] Environment variables documented if new ones added
- [ ] Feature can be disabled via configuration if needed

---

## Anti-Patterns to Avoid

- ❌ Don't create new URL filtering logic - reuse existing `base_prefix` patterns
- ❌ Don't skip graceful fallback - SPA features must be additive only
- ❌ Don't use Unicode characters in console output (Windows compatibility)
- ❌ Don't instantiate crawler directly - use context access pattern
- ❌ Don't hardcode framework patterns - use configurable detection thresholds
- ❌ Don't break existing test cases - only add new tests
- ❌ Don't ignore performance impact - validate <10% overhead for non-SPA sites

---

**Confidence Score**: 9/10 for one-pass implementation success

**Rationale**: Comprehensive codebase analysis, external research, specific implementation patterns, and extensive validation strategy provide all necessary context for successful implementation. The 1-point deduction accounts for potential edge cases in real-world SPA variations that may require minor adjustments.