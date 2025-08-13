# SPA Crawling Implementation Reference

A comprehensive reference for implementing Single Page Application (SPA) crawling with BeautifulSoup, urllib.parse, Crawl4AI, and async patterns.

## Table of Contents

1. [BeautifulSoup CSS Selector Patterns](#beautifulsoup-css-selector-patterns)
2. [urllib.parse URL Manipulation](#urlparse-url-manipulation)
3. [Crawl4AI JavaScript Execution](#crawl4ai-javascript-execution)
4. [Async/Await Concurrency Patterns](#asyncawait-concurrency-patterns)
5. [SPA-Specific Implementation Patterns](#spa-specific-implementation-patterns)
6. [Error Handling and Edge Cases](#error-handling-and-edge-cases)
7. [Performance Optimization](#performance-optimization)

## BeautifulSoup CSS Selector Patterns

### Navigation Element Selection

**2024 Best Practices for SPA Navigation:**

```python
from bs4 import BeautifulSoup
import re

# Modern CSS selector patterns for navigation elements
def extract_navigation_elements(soup):
    """Extract navigation elements using multiple fallback selectors."""
    
    # Priority-ordered selectors for navigation menus
    nav_selectors = [
        'nav[data-testid*="navigation"]',  # Modern data-testid patterns
        'nav.main-nav, nav.navbar, nav.menu',  # Common class patterns
        '[role="navigation"]',  # ARIA role-based selection
        'header nav, .header nav',  # Header-based navigation
        '.sidebar nav, aside nav',  # Sidebar navigation
        'ul.menu, ul.nav, .menu-list',  # List-based menus
        '[data-component="Navigation"]',  # Component-based patterns
    ]
    
    for selector in nav_selectors:
        elements = soup.select(selector)
        if elements:
            return elements
    
    return []

# Extract navigation links with advanced filtering
def extract_nav_links(soup):
    """Extract navigation links with SPA-specific patterns."""
    
    # SPA-specific link patterns
    spa_link_selectors = [
        'a[href^="#"]',  # Hash-based routes
        'a[href^="/"]',  # Absolute paths
        'a[data-route]',  # Route data attributes
        'button[data-navigate]',  # Button-based navigation
        '[onclick*="navigate"], [onclick*="router"]',  # JS navigation
    ]
    
    links = []
    for selector in spa_link_selectors:
        found_links = soup.select(selector)
        for link in found_links:
            href = link.get('href') or link.get('data-route') or link.get('data-navigate')
            text = link.get_text(strip=True)
            if href and text:
                links.append({
                    'url': href,
                    'text': text,
                    'element': link.name,
                    'classes': link.get('class', [])
                })
    
    return links
```

### Regex Patterns for URL Extraction

```python
import re

# SPA URL pattern matchers
SPA_URL_PATTERNS = {
    'hash_routes': re.compile(r'#/([\w\-/]+)'),
    'fragment_ids': re.compile(r'#([\w\-]+)$'),
    'spa_paths': re.compile(r'/app/([\w\-/]+)'),
    'query_routes': re.compile(r'\?route=([\w\-/]+)'),
    'api_endpoints': re.compile(r'/api/v\d+/([\w\-/]+)'),
}

def extract_spa_urls(html_content):
    """Extract SPA-specific URLs using regex patterns."""
    
    urls = {}
    for pattern_name, pattern in SPA_URL_PATTERNS.items():
        matches = pattern.findall(html_content)
        urls[pattern_name] = list(set(matches))  # Remove duplicates
    
    return urls

# Advanced URL extraction with context
def extract_urls_with_context(soup):
    """Extract URLs with surrounding context for better understanding."""
    
    url_context = []
    
    # Find all elements with href attributes
    for element in soup.find_all(href=True):
        href = element['href']
        
        # Get parent context
        parent = element.parent
        parent_text = parent.get_text(strip=True) if parent else ""
        
        # Get surrounding text context
        prev_sibling = element.previous_sibling
        next_sibling = element.next_sibling
        
        context = {
            'url': href,
            'text': element.get_text(strip=True),
            'parent_context': parent_text[:100],
            'tag': element.name,
            'classes': element.get('class', []),
            'data_attrs': {k: v for k, v in element.attrs.items() if k.startswith('data-')},
        }
        
        url_context.append(context)
    
    return url_context
```

### Error Handling for Malformed HTML

```python
def robust_parsing(html_content, encoding='utf-8'):
    """Parse HTML with robust error handling."""
    
    # Parser fallback chain
    parsers = ['lxml', 'html.parser', 'html5lib']
    
    for parser in parsers:
        try:
            soup = BeautifulSoup(html_content, parser, from_encoding=encoding)
            
            # Validate parsing success
            if soup.title or soup.find('body') or soup.find_all():
                return soup, parser
                
        except Exception as e:
            print(f"Parser {parser} failed: {e}")
            continue
    
    # Fallback: clean HTML and retry
    try:
        # Remove problematic characters
        cleaned_html = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', html_content)
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        return soup, 'html.parser (cleaned)'
    except Exception as e:
        raise ValueError(f"All parsers failed: {e}")

# Defensive element extraction
def safe_extract_text(element, default=""):
    """Safely extract text from potentially malformed elements."""
    
    if not element:
        return default
    
    try:
        # Multiple extraction strategies
        text = element.get_text(strip=True)
        if not text and element.string:
            text = str(element.string).strip()
        if not text and element.contents:
            text = ' '.join(str(content).strip() for content in element.contents 
                          if hasattr(content, 'strip'))
        
        return text or default
        
    except Exception:
        return default

def safe_extract_attr(element, attr, default=None):
    """Safely extract attributes with fallbacks."""
    
    if not element:
        return default
    
    try:
        value = element.get(attr, default)
        # Handle list attributes (like class)
        if isinstance(value, list):
            return ' '.join(value)
        return value
    except Exception:
        return default
```

## urllib.parse URL Manipulation

### SPA URL Processing Functions

```python
from urllib.parse import urlparse, urljoin, urldefrag, parse_qs, urlencode
from urllib.parse import quote, unquote

def normalize_spa_url(url, base_url=None):
    """Normalize SPA URLs for consistent processing."""
    
    if not url:
        return None
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Handle relative URLs
    if base_url and not parsed.netloc:
        url = urljoin(base_url, url)
        parsed = urlparse(url)
    
    # Normalize scheme (default to https)
    scheme = parsed.scheme.lower() if parsed.scheme else 'https'
    
    # Normalize netloc (convert to lowercase)
    netloc = parsed.netloc.lower()
    
    # Clean path (remove double slashes, ensure leading slash)
    path = parsed.path
    if path:
        path = re.sub(r'/+', '/', path)
        if not path.startswith('/'):
            path = '/' + path
    else:
        path = '/'
    
    # Preserve query and fragment for SPAs
    query = parsed.query
    fragment = parsed.fragment
    
    # Reconstruct normalized URL
    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized += f"?{query}"
    if fragment:
        normalized += f"#{fragment}"
    
    return normalized

def extract_spa_route_info(url):
    """Extract routing information from SPA URLs."""
    
    parsed = urlparse(url)
    
    route_info = {
        'base_url': f"{parsed.scheme}://{parsed.netloc}",
        'path': parsed.path,
        'query_params': parse_qs(parsed.query),
        'fragment': parsed.fragment,
        'is_hash_route': bool(parsed.fragment and '/' in parsed.fragment),
        'is_query_route': 'route' in parse_qs(parsed.query),
    }
    
    # Extract hash-based route
    if route_info['is_hash_route']:
        fragment = parsed.fragment
        if fragment.startswith('/'):
            route_info['hash_route'] = fragment
        elif '/' in fragment:
            route_info['hash_route'] = '/' + fragment.split('/', 1)[1]
    
    # Extract query-based route
    if route_info['is_query_route']:
        query_params = parse_qs(parsed.query)
        if 'route' in query_params:
            route_info['query_route'] = query_params['route'][0]
    
    return route_info

def build_spa_url(base_url, route=None, query_params=None, fragment=None):
    """Build SPA URLs with proper encoding."""
    
    parsed_base = urlparse(base_url)
    
    # Start with base components
    scheme = parsed_base.scheme or 'https'
    netloc = parsed_base.netloc
    path = parsed_base.path or '/'
    
    # Add route to path if provided
    if route:
        if not route.startswith('/'):
            route = '/' + route
        path = path.rstrip('/') + route
    
    # Build query string
    query = ''
    if query_params:
        query = urlencode(query_params, doseq=True)
    
    # Build fragment
    fragment_str = ''
    if fragment:
        fragment_str = quote(fragment, safe='/')
    
    # Construct final URL
    url = f"{scheme}://{netloc}{path}"
    if query:
        url += f"?{query}"
    if fragment_str:
        url += f"#{fragment_str}"
    
    return url
```

### URL Validation and Sanitization

```python
import re
from urllib.parse import urlparse

def validate_spa_url(url, allowed_schemes=None, allowed_domains=None):
    """Validate and sanitize SPA URLs."""
    
    if not url or not isinstance(url, str):
        return False, "Invalid URL format"
    
    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"URL parsing failed: {e}"
    
    # Check scheme
    allowed_schemes = allowed_schemes or ['http', 'https']
    if parsed.scheme and parsed.scheme.lower() not in allowed_schemes:
        return False, f"Scheme {parsed.scheme} not allowed"
    
    # Check domain if specified
    if allowed_domains and parsed.netloc:
        domain_match = any(
            parsed.netloc.lower().endswith(domain.lower()) 
            for domain in allowed_domains
        )
        if not domain_match:
            return False, f"Domain {parsed.netloc} not allowed"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'file:',
        r'ftp:',
    ]
    
    url_lower = url.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, url_lower):
            return False, f"Suspicious pattern detected: {pattern}"
    
    return True, "Valid URL"

def sanitize_url_fragment(fragment):
    """Sanitize URL fragments for safe processing."""
    
    if not fragment:
        return ""
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>"\'\x00-\x1f\x7f-\x9f]', '', fragment)
    
    # Limit length
    if len(sanitized) > 500:
        sanitized = sanitized[:500]
    
    return sanitized
```

## Crawl4AI JavaScript Execution

### SPA-Specific Crawl Configurations

```python
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

async def crawl_spa_with_js(url, wait_strategy="dom_ready", timeout=30):
    """Crawl SPA with JavaScript execution and smart waiting."""
    
    browser_config = BrowserConfig(
        headless=True,
        browser_type="chromium",
        viewport_width=1920,
        viewport_height=1080,
        user_agent="Mozilla/5.0 (compatible; SPACrawler/1.0)"
    )
    
    # SPA-optimized crawl configuration
    crawl_config = CrawlerRunConfig(
        # JavaScript execution
        execute_js=True,
        js_on_dom_content_loaded=True,
        
        # Wait strategies for SPAs
        wait_for_selector="[data-testid='app-loaded'], .app-content, main",
        wait_for_timeout=5000,  # 5 seconds
        
        # Performance settings
        cache_mode="enabled",
        timeout=timeout,
        
        # Content extraction
        scraping_strategy=LXMLWebScrapingStrategy(),
        
        # Network settings
        simulate_user=True,
        magic=True,  # Enable smart waiting
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            result = await crawler.arun(url=url, config=crawl_config)
            return result
        except Exception as e:
            print(f"Crawling failed for {url}: {e}")
            return None

# Advanced JavaScript execution for SPA content loading
async def execute_spa_navigation_js(crawler, url, navigation_actions=None):
    """Execute JavaScript to navigate SPA and load content."""
    
    # Default navigation actions for SPAs
    if navigation_actions is None:
        navigation_actions = [
            "window.scrollTo(0, document.body.scrollHeight);",  # Scroll to trigger lazy loading
            "document.querySelectorAll('button[data-load-more]').forEach(btn => btn.click());",  # Click load more
            "await new Promise(resolve => setTimeout(resolve, 2000));",  # Wait for content
        ]
    
    # JavaScript to execute before content extraction
    spa_js_code = """
    // Wait for SPA initialization
    async function waitForSPAReady() {
        // Wait for common SPA ready indicators
        const selectors = [
            '[data-testid="app-ready"]',
            '.app-loaded',
            '[data-app-status="ready"]',
            'main[data-loaded="true"]'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) return true;
        }
        
        // Wait for React/Vue/Angular to be ready
        if (window.React || window.Vue || window.ng) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            return true;
        }
        
        // Fallback: wait for DOM mutations to settle
        return new Promise(resolve => {
            let timeout;
            const observer = new MutationObserver(() => {
                clearTimeout(timeout);
                timeout = setTimeout(() => {
                    observer.disconnect();
                    resolve(true);
                }, 500);
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            // Maximum wait time
            setTimeout(() => {
                observer.disconnect();
                resolve(true);
            }, 10000);
        });
    }
    
    // Execute navigation actions
    await waitForSPAReady();
    """ + "\n".join(navigation_actions)
    
    config = CrawlerRunConfig(
        js_code=spa_js_code,
        wait_for_timeout=10000,
        execute_js=True,
    )
    
    return await crawler.arun(url=url, config=config)
```

### JavaScript Wait Strategies

```python
# Different wait strategies for various SPA scenarios
SPA_WAIT_STRATEGIES = {
    "react_app": {
        "wait_for_selector": "[data-reactroot], #root > div",
        "js_code": """
        // Wait for React to finish rendering
        function waitForReact() {
            return new Promise(resolve => {
                if (window.React && window.React.version) {
                    // React DevTools available
                    const checkReactReady = () => {
                        const rootElement = document.querySelector('#root, [data-reactroot]');
                        if (rootElement && rootElement.children.length > 0) {
                            resolve();
                        } else {
                            setTimeout(checkReactReady, 100);
                        }
                    };
                    checkReactReady();
                } else {
                    // Fallback to DOM observation
                    setTimeout(resolve, 2000);
                }
            });
        }
        await waitForReact();
        """
    },
    
    "vue_app": {
        "wait_for_selector": "[data-v-], .vue-app",
        "js_code": """
        // Wait for Vue.js to finish mounting
        function waitForVue() {
            return new Promise(resolve => {
                if (window.Vue) {
                    setTimeout(resolve, 1000);
                } else {
                    const checkVueReady = () => {
                        const vueElements = document.querySelectorAll('[data-v-]');
                        if (vueElements.length > 0) {
                            resolve();
                        } else {
                            setTimeout(checkVueReady, 100);
                        }
                    };
                    checkVueReady();
                }
            });
        }
        await waitForVue();
        """
    },
    
    "angular_app": {
        "wait_for_selector": "app-root, [ng-app]",
        "js_code": """
        // Wait for Angular to finish bootstrapping
        function waitForAngular() {
            return new Promise(resolve => {
                if (window.ng && window.ng.probe) {
                    // Angular DevTools available
                    setTimeout(resolve, 1000);
                } else {
                    // Wait for app-root or ng-app
                    const checkAngularReady = () => {
                        const appRoot = document.querySelector('app-root, [ng-app]');
                        if (appRoot && appRoot.children.length > 0) {
                            resolve();
                        } else {
                            setTimeout(checkAngularReady, 100);
                        }
                    };
                    checkAngularReady();
                }
            });
        }
        await waitForAngular();
        """
    },
    
    "generic_spa": {
        "wait_for_selector": "main, .app, .content",
        "js_code": """
        // Generic SPA waiting strategy
        function waitForContent() {
            return new Promise(resolve => {
                let attempts = 0;
                const maxAttempts = 50;
                
                const checkContent = () => {
                    attempts++;
                    
                    // Check for common content indicators
                    const contentElements = document.querySelectorAll('main, .app, .content, [data-testid]');
                    const hasText = document.body.innerText.trim().length > 100;
                    
                    if ((contentElements.length > 0 && hasText) || attempts >= maxAttempts) {
                        resolve();
                    } else {
                        setTimeout(checkContent, 200);
                    }
                };
                
                checkContent();
            });
        }
        await waitForContent();
        """
    }
}

async def crawl_spa_with_strategy(url, strategy_name="generic_spa"):
    """Crawl SPA using predefined strategy."""
    
    strategy = SPA_WAIT_STRATEGIES.get(strategy_name, SPA_WAIT_STRATEGIES["generic_spa"])
    
    config = CrawlerRunConfig(
        wait_for_selector=strategy["wait_for_selector"],
        js_code=strategy["js_code"],
        execute_js=True,
        wait_for_timeout=15000,
        simulate_user=True,
    )
    
    browser_config = BrowserConfig(headless=True)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        return await crawler.arun(url=url, config=config)
```

## Async/Await Concurrency Patterns

### Semaphore-Based Rate Limiting

```python
import asyncio
import time
from typing import List, Callable, Any

class RateLimitedCrawler:
    """Rate-limited crawler with proper error handling."""
    
    def __init__(self, max_concurrent=5, delay_between_requests=1.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay_between_requests
        self.last_request_time = 0
    
    async def crawl_with_limit(self, url: str, crawl_func: Callable) -> Any:
        """Crawl a single URL with rate limiting."""
        
        async with self.semaphore:
            # Enforce minimum delay between requests
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.delay:
                await asyncio.sleep(self.delay - time_since_last)
            
            self.last_request_time = time.time()
            
            try:
                result = await crawl_func(url)
                return {"url": url, "success": True, "data": result}
            except Exception as e:
                return {"url": url, "success": False, "error": str(e)}
    
    async def crawl_multiple(self, urls: List[str], crawl_func: Callable) -> List[dict]:
        """Crawl multiple URLs concurrently with rate limiting."""
        
        tasks = [self.crawl_with_limit(url, crawl_func) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions from gather
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "url": "unknown",
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results

# Advanced concurrency control with backoff
class AdaptiveCrawler:
    """Crawler with adaptive rate limiting and error handling."""
    
    def __init__(self, initial_concurrency=5, max_retries=3):
        self.initial_concurrency = initial_concurrency
        self.current_concurrency = initial_concurrency
        self.max_retries = max_retries
        self.error_count = 0
        self.success_count = 0
        
    async def adaptive_crawl(self, urls: List[str], crawl_func: Callable) -> List[dict]:
        """Crawl with adaptive concurrency based on error rates."""
        
        # Adjust concurrency based on recent performance
        if self.error_count > self.success_count and self.current_concurrency > 1:
            self.current_concurrency = max(1, self.current_concurrency // 2)
        elif self.success_count > self.error_count * 2:
            self.current_concurrency = min(20, self.current_concurrency + 1)
        
        semaphore = asyncio.Semaphore(self.current_concurrency)
        
        async def crawl_with_retry(url: str) -> dict:
            """Crawl with retry logic and backoff."""
            
            for attempt in range(self.max_retries + 1):
                async with semaphore:
                    try:
                        if attempt > 0:
                            # Exponential backoff
                            await asyncio.sleep(2 ** attempt)
                        
                        result = await crawl_func(url)
                        self.success_count += 1
                        return {"url": url, "success": True, "data": result, "attempts": attempt + 1}
                        
                    except asyncio.TimeoutError:
                        if attempt == self.max_retries:
                            self.error_count += 1
                            return {"url": url, "success": False, "error": "Timeout", "attempts": attempt + 1}
                    except Exception as e:
                        if attempt == self.max_retries:
                            self.error_count += 1
                            return {"url": url, "success": False, "error": str(e), "attempts": attempt + 1}
        
        tasks = [crawl_with_retry(url) for url in urls]
        return await asyncio.gather(*tasks)

# Practical usage example
async def crawl_spa_urls_concurrently(urls: List[str], max_concurrent=5):
    """Crawl SPA URLs with proper concurrency control."""
    
    crawler = RateLimitedCrawler(max_concurrent=max_concurrent, delay_between_requests=0.5)
    
    async def spa_crawl_func(url: str):
        """Single URL crawling function."""
        return await crawl_spa_with_js(url, wait_strategy="dom_ready", timeout=30)
    
    results = await crawler.crawl_multiple(urls, spa_crawl_func)
    
    # Process results
    successful_crawls = [r for r in results if r["success"]]
    failed_crawls = [r for r in results if not r["success"]]
    
    print(f"Successful: {len(successful_crawls)}, Failed: {len(failed_crawls)}")
    
    return successful_crawls, failed_crawls
```

### Error Handling and Recovery

```python
import asyncio
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

class CrawlErrorHandler:
    """Centralized error handling for crawling operations."""
    
    RECOVERABLE_ERRORS = [
        "TimeoutError",
        "ConnectionError",
        "HTTPError",
        "NetworkError"
    ]
    
    NON_RECOVERABLE_ERRORS = [
        "AuthenticationError",
        "PermissionError",
        "InvalidURLError"
    ]
    
    @staticmethod
    def is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        error_name = type(error).__name__
        return error_name in CrawlErrorHandler.RECOVERABLE_ERRORS
    
    @staticmethod
    async def handle_crawl_error(url: str, error: Exception, attempt: int) -> Optional[dict]:
        """Handle crawling errors with appropriate responses."""
        
        error_info = {
            "url": url,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempt": attempt,
            "recoverable": CrawlErrorHandler.is_recoverable(error)
        }
        
        logger.warning(f"Crawl error for {url}: {error_info}")
        
        if error_info["recoverable"]:
            # Calculate retry delay
            delay = min(60, 2 ** attempt)  # Exponential backoff, max 60s
            logger.info(f"Retrying {url} in {delay}s (attempt {attempt})")
            await asyncio.sleep(delay)
            return None  # Signal retry
        else:
            logger.error(f"Non-recoverable error for {url}: {error}")
            return error_info

# Robust gather implementation
async def robust_gather(*tasks, return_exceptions=True, max_retries=3):
    """Enhanced gather with better error handling."""
    
    if not return_exceptions:
        # Use standard gather for non-exception handling
        return await asyncio.gather(*tasks)
    
    results = []
    
    for task in tasks:
        attempt = 0
        while attempt <= max_retries:
            try:
                if asyncio.iscoroutine(task):
                    result = await task
                else:
                    result = await task
                
                results.append(result)
                break
                
            except Exception as e:
                attempt += 1
                
                if attempt > max_retries:
                    results.append(e)
                    break
                
                # Handle recoverable errors
                if CrawlErrorHandler.is_recoverable(e):
                    delay = 2 ** (attempt - 1)
                    await asyncio.sleep(delay)
                else:
                    results.append(e)
                    break
    
    return results

# Context manager for crawling sessions
class CrawlingSession:
    """Context manager for crawling operations with cleanup."""
    
    def __init__(self, crawler_config=None):
        self.crawler = None
        self.crawler_config = crawler_config or {}
        self.active_tasks = set()
    
    async def __aenter__(self):
        self.crawler = AsyncWebCrawler(config=self.crawler_config)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel any remaining tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        # Clean up crawler
        if self.crawler:
            await self.crawler.aclose()
    
    async def crawl_url(self, url: str, config=None) -> dict:
        """Crawl a URL within the session."""
        
        task = asyncio.create_task(self.crawler.arun(url, config=config))
        self.active_tasks.add(task)
        
        try:
            result = await task
            return {"url": url, "success": True, "data": result}
        except Exception as e:
            return {"url": url, "success": False, "error": str(e)}
        finally:
            self.active_tasks.discard(task)

# Usage example
async def crawl_with_robust_error_handling(urls: List[str]):
    """Example of robust crawling with proper error handling."""
    
    async with CrawlingSession() as session:
        # Create tasks for all URLs
        tasks = [session.crawl_url(url) for url in urls]
        
        # Execute with robust error handling
        results = await robust_gather(*tasks, max_retries=2)
        
        return results
```

## SPA-Specific Implementation Patterns

### Complete SPA Crawling Workflow

```python
import asyncio
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
import re

class SPACrawler:
    """Specialized crawler for Single Page Applications."""
    
    def __init__(self, base_url: str, max_depth: int = 3, max_pages: int = 100):
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.discovered_routes: Set[str] = set()
        self.crawl_results: List[Dict] = []
    
    async def discover_spa_routes(self, initial_content: str) -> Set[str]:
        """Discover SPA routes from initial page content."""
        
        routes = set()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(initial_content, 'lxml')
        
        # Extract hash routes
        hash_routes = soup.find_all(href=re.compile(r'^#/'))
        for link in hash_routes:
            href = link.get('href')
            if href and href.startswith('#/'):
                routes.add(href[1:])  # Remove # prefix
        
        # Extract data-route attributes
        route_elements = soup.find_all(attrs={'data-route': True})
        for element in route_elements:
            route = element.get('data-route')
            if route:
                routes.add(route)
        
        # Extract JavaScript-defined routes
        script_tags = soup.find_all('script')
        for script in script_tags:
            if script.string:
                # Look for route definitions in JavaScript
                route_patterns = [
                    r"route[s]?\s*:\s*['\"]([^'\"]+)['\"]",
                    r"path\s*:\s*['\"]([^'\"]+)['\"]",
                    r"['\"]/([\w\-/]+)['\"]",
                ]
                
                for pattern in route_patterns:
                    matches = re.findall(pattern, script.string)
                    for match in matches:
                        if '/' in match:
                            routes.add(match)
        
        return routes
    
    async def crawl_spa_page(self, url: str, depth: int = 0) -> Dict:
        """Crawl a single SPA page with route discovery."""
        
        if depth > self.max_depth or len(self.visited_urls) >= self.max_pages:
            return None
        
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        try:
            # Crawl with SPA-specific configuration
            result = await crawl_spa_with_js(url, wait_strategy="dom_ready")
            
            if not result or not result.success:
                return {"url": url, "success": False, "error": "Crawl failed"}
            
            # Discover new routes from content
            if result.html:
                new_routes = await self.discover_spa_routes(result.html)
                self.discovered_routes.update(new_routes)
            
            crawl_data = {
                "url": url,
                "success": True,
                "content": result.markdown or "",
                "html": result.html or "",
                "depth": depth,
                "discovered_routes": list(new_routes) if 'new_routes' in locals() else [],
                "links": getattr(result, 'links', {}),
            }
            
            self.crawl_results.append(crawl_data)
            return crawl_data
            
        except Exception as e:
            error_data = {"url": url, "success": False, "error": str(e), "depth": depth}
            self.crawl_results.append(error_data)
            return error_data
    
    async def crawl_spa_breadth_first(self) -> List[Dict]:
        """Crawl SPA using breadth-first approach with route discovery."""
        
        # Start with base URL
        current_level = [self.base_url]
        
        for depth in range(self.max_depth + 1):
            if not current_level or len(self.visited_urls) >= self.max_pages:
                break
            
            # Crawl all URLs at current depth
            level_tasks = [
                self.crawl_spa_page(url, depth) 
                for url in current_level[:10]  # Limit per level
            ]
            
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Prepare next level URLs
            next_level = []
            
            # Add discovered routes to next level
            for route in self.discovered_routes:
                full_url = urljoin(self.base_url, route)
                if full_url not in self.visited_urls:
                    next_level.append(full_url)
            
            current_level = next_level
            
            # Clear discovered routes for next iteration
            self.discovered_routes.clear()
        
        return [r for r in self.crawl_results if r is not None]

# Usage example
async def crawl_complete_spa(base_url: str, max_pages: int = 50):
    """Complete SPA crawling example."""
    
    crawler = SPACrawler(base_url, max_depth=3, max_pages=max_pages)
    results = await crawler.crawl_spa_breadth_first()
    
    # Process results
    successful_pages = [r for r in results if r.get("success")]
    failed_pages = [r for r in results if not r.get("success")]
    
    print(f"Successfully crawled {len(successful_pages)} pages")
    print(f"Failed to crawl {len(failed_pages)} pages")
    
    return {
        "successful": successful_pages,
        "failed": failed_pages,
        "total_discovered_urls": len(crawler.visited_urls),
        "crawl_summary": {
            "base_url": base_url,
            "max_depth": crawler.max_depth,
            "pages_crawled": len(results),
            "success_rate": len(successful_pages) / len(results) if results else 0
        }
    }
```

## Error Handling and Edge Cases

### Common SPA Crawling Issues

```python
class SPAErrorHandler:
    """Handle common SPA crawling errors and edge cases."""
    
    @staticmethod
    async def handle_infinite_scroll(crawler, url, max_scrolls=10):
        """Handle infinite scroll SPAs."""
        
        js_scroll_code = f"""
        let scrollCount = 0;
        const maxScrolls = {max_scrolls};
        
        async function scrollAndWait() {{
            while (scrollCount < maxScrolls) {{
                const beforeHeight = document.body.scrollHeight;
                window.scrollTo(0, document.body.scrollHeight);
                
                // Wait for new content to load
                await new Promise(resolve => setTimeout(resolve, 2000));
                
                const afterHeight = document.body.scrollHeight;
                
                // Stop if no new content loaded
                if (beforeHeight === afterHeight) {{
                    break;
                }}
                
                scrollCount++;
            }}
        }}
        
        await scrollAndWait();
        """
        
        config = CrawlerRunConfig(
            js_code=js_scroll_code,
            wait_for_timeout=30000,
            execute_js=True,
        )
        
        return await crawler.arun(url=url, config=config)
    
    @staticmethod
    async def handle_lazy_loading(crawler, url, selectors_to_wait=None):
        """Handle lazy-loaded content in SPAs."""
        
        if selectors_to_wait is None:
            selectors_to_wait = [
                '.lazy-load-complete',
                '[data-loaded="true"]',
                '.content-loaded'
            ]
        
        wait_selectors = ', '.join(selectors_to_wait)
        
        js_lazy_load_code = f"""
        // Trigger lazy loading by scrolling and clicking load buttons
        function triggerLazyLoading() {{
            // Scroll to trigger intersection observers
            const scrollableElements = document.querySelectorAll('[data-lazy], .lazy, [loading="lazy"]');
            scrollableElements.forEach(el => {{
                el.scrollIntoView();
            }});
            
            // Click any "load more" buttons
            const loadButtons = document.querySelectorAll(
                'button[data-load-more], .load-more, [data-action="load"]'
            );
            loadButtons.forEach(btn => btn.click());
        }}
        
        // Wait for content to load
        function waitForLazyContent() {{
            return new Promise(resolve => {{
                const checkLoaded = () => {{
                    const indicators = document.querySelectorAll('{wait_selectors}');
                    if (indicators.length > 0) {{
                        resolve();
                    }} else {{
                        setTimeout(checkLoaded, 500);
                    }}
                }};
                
                setTimeout(resolve, 10000); // Max wait time
                checkLoaded();
            }});
        }}
        
        triggerLazyLoading();
        await waitForLazyContent();
        """
        
        config = CrawlerRunConfig(
            js_code=js_lazy_load_code,
            wait_for_timeout=15000,
            execute_js=True,
        )
        
        return await crawler.arun(url=url, config=config)
    
    @staticmethod
    def handle_malformed_spa_content(content: str) -> str:
        """Clean and fix malformed SPA content."""
        
        if not content:
            return ""
        
        # Remove problematic characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        # Fix common SPA artifacts
        artifacts_to_remove = [
            r'<script[^>]*>.*?</script>',  # Remove script tags
            r'<style[^>]*>.*?</style>',   # Remove style tags
            r'<!--.*?-->',                # Remove comments
            r'<noscript[^>]*>.*?</noscript>',  # Remove noscript
        ]
        
        for pattern in artifacts_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        return cleaned.strip()

# Edge case handlers
async def handle_spa_authentication_required(crawler, url, login_data=None):
    """Handle SPAs that require authentication."""
    
    if not login_data:
        return {"error": "Authentication required but no login data provided"}
    
    # First, check if login is needed
    initial_result = await crawler.arun(url=url)
    
    if "login" in initial_result.html.lower() or "signin" in initial_result.html.lower():
        # Perform login
        login_js = f"""
        // Fill login form
        const emailField = document.querySelector('input[type="email"], input[name="email"], #email');
        const passwordField = document.querySelector('input[type="password"], input[name="password"], #password');
        const loginButton = document.querySelector('button[type="submit"], .login-btn, #login');
        
        if (emailField && passwordField && loginButton) {{
            emailField.value = '{login_data.get("email", "")}';
            passwordField.value = '{login_data.get("password", "")}';
            loginButton.click();
            
            // Wait for login to complete
            await new Promise(resolve => setTimeout(resolve, 3000));
        }}
        """
        
        config = CrawlerRunConfig(
            js_code=login_js,
            wait_for_timeout=10000,
            execute_js=True,
        )
        
        # Attempt login and re-crawl
        await crawler.arun(url=url, config=config)
        return await crawler.arun(url=url)  # Crawl again after login
    
    return initial_result

async def handle_spa_rate_limiting(crawler, url, max_retries=5):
    """Handle rate limiting from SPA endpoints."""
    
    for attempt in range(max_retries):
        try:
            result = await crawler.arun(url=url)
            
            # Check for rate limiting indicators
            if result.html and ("rate limit" in result.html.lower() or 
                              "too many requests" in result.html.lower()):
                # Exponential backoff
                delay = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"Rate limited, waiting {delay}s before retry {attempt + 1}")
                await asyncio.sleep(delay)
                continue
            
            return result
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = 2 ** attempt * 5
                await asyncio.sleep(delay)
                continue
            else:
                raise e
    
    return {"error": "Max retries exceeded due to rate limiting"}
```

## Performance Optimization

### Caching and Optimization Strategies

```python
import hashlib
import json
import aiofiles
from pathlib import Path
from typing import Optional

class SPACrawlCache:
    """Caching system for SPA crawling results."""
    
    def __init__(self, cache_dir: str = "./spa_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def _get_cache_key(self, url: str, config_hash: str = "") -> str:
        """Generate cache key for URL and configuration."""
        content = f"{url}:{config_hash}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_cached_result(self, url: str, config_hash: str = "") -> Optional[dict]:
        """Retrieve cached crawl result."""
        
        cache_key = self._get_cache_key(url, config_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            async with aiofiles.open(cache_file, 'r') as f:
                data = json.loads(await f.read())
            
            # Check TTL
            import time
            if time.time() - data.get('timestamp', 0) > self.ttl_seconds:
                cache_file.unlink()  # Remove expired cache
                return None
            
            return data.get('result')
            
        except Exception:
            return None
    
    async def cache_result(self, url: str, result: dict, config_hash: str = ""):
        """Cache crawl result."""
        
        cache_key = self._get_cache_key(url, config_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            'url': url,
            'timestamp': time.time(),
            'result': result,
            'config_hash': config_hash
        }
        
        try:
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, default=str))
        except Exception as e:
            print(f"Failed to cache result for {url}: {e}")

# Optimized crawler with caching
class OptimizedSPACrawler:
    """Optimized SPA crawler with caching and resource management."""
    
    def __init__(self, cache_ttl_hours: int = 24, max_concurrent: int = 5):
        self.cache = SPACrawlCache(ttl_hours=cache_ttl_hours)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'errors': 0
        }
    
    async def crawl_with_cache(self, url: str, config: CrawlerRunConfig = None) -> dict:
        """Crawl URL with caching support."""
        
        # Generate config hash for cache key
        config_hash = ""
        if config:
            config_str = json.dumps(config.__dict__, sort_keys=True, default=str)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Check cache first
        cached_result = await self.cache.get_cached_result(url, config_hash)
        if cached_result:
            self.session_stats['cache_hits'] += 1
            return cached_result
        
        self.session_stats['cache_misses'] += 1
        self.session_stats['total_requests'] += 1
        
        # Crawl with rate limiting
        async with self.semaphore:
            try:
                browser_config = BrowserConfig(headless=True)
                async with AsyncWebCrawler(config=browser_config) as crawler:
                    result = await crawler.arun(url=url, config=config)
                
                # Prepare result for caching
                cache_result = {
                    'url': url,
                    'success': result.success if result else False,
                    'content': result.markdown if result and result.success else "",
                    'html': result.html if result and result.success else "",
                    'links': getattr(result, 'links', {}) if result else {},
                    'error': result.error_message if result and not result.success else None
                }
                
                # Cache successful results
                if cache_result['success']:
                    await self.cache.cache_result(url, cache_result, config_hash)
                
                return cache_result
                
            except Exception as e:
                self.session_stats['errors'] += 1
                error_result = {
                    'url': url,
                    'success': False,
                    'error': str(e),
                    'content': "",
                    'html': "",
                    'links': {}
                }
                return error_result
    
    def get_cache_stats(self) -> dict:
        """Get caching performance statistics."""
        
        total_attempts = self.session_stats['cache_hits'] + self.session_stats['cache_misses']
        cache_hit_rate = (self.session_stats['cache_hits'] / total_attempts * 100 
                         if total_attempts > 0 else 0)
        
        return {
            **self.session_stats,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'total_attempts': total_attempts
        }

# Usage example
async def optimized_spa_crawling_example():
    """Example of optimized SPA crawling with caching."""
    
    urls = [
        "https://example-spa.com/",
        "https://example-spa.com/about",
        "https://example-spa.com/products",
        "https://example-spa.com/contact"
    ]
    
    crawler = OptimizedSPACrawler(cache_ttl_hours=12, max_concurrent=3)
    
    # Configure for SPA crawling
    spa_config = CrawlerRunConfig(
        execute_js=True,
        wait_for_selector="main, .app-content",
        wait_for_timeout=10000,
        js_code="await new Promise(resolve => setTimeout(resolve, 2000));"  # Wait for SPA
    )
    
    # Crawl all URLs
    tasks = [crawler.crawl_with_cache(url, spa_config) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Print statistics
    stats = crawler.get_cache_stats()
    print(f"Crawling completed:")
    print(f"  Cache hit rate: {stats['cache_hit_rate_percent']}%")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Errors: {stats['errors']}")
    
    # Process results
    successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
    return successful_results, stats
```

---

This comprehensive reference document provides practical, real-world patterns for implementing SPA crawling with the specific libraries mentioned. Each section includes working code examples, error handling patterns, and performance optimizations based on current best practices from 2024.

The patterns focus on:
- Robust CSS selector strategies for navigation elements
- Safe URL manipulation and validation techniques  
- Advanced JavaScript execution patterns for SPAs
- Production-ready async/await concurrency management
- Complete error handling and edge case management
- Performance optimization through caching and resource management

All code examples are designed to be immediately usable and include proper error handling for production environments.