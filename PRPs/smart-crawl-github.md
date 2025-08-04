# PRP: Smart Crawl GitHub Tool

name: "Smart Crawl GitHub - Repository Markdown Indexing Tool"
description: |
  MCP server tool for cloning GitHub repositories, extracting markdown files, 
  and storing them in the vector database following existing patterns.

## Goal

Create a new MCP server tool called `smart_crawl_github` that clones GitHub repositories to temporary folders, extracts all markdown files, processes them using the same chunking and storage patterns as `smart_crawl_url`, and stores them in the vector database with appropriate source naming for easy retrieval.

## Why

- **Repository Documentation Access**: Enable RAG queries against entire GitHub repository documentation
- **Knowledge Base Expansion**: Add GitHub repositories as sources alongside web content  
- **Developer Productivity**: Allow AI assistants to answer questions about specific project documentation
- **Consistent Data Processing**: Apply proven patterns from `smart_crawl_url` to repository content
- **Source Organization**: Use GitHub-specific source naming for easy filtering and discovery

## What

An MCP server tool that:
1. **Validates and clones** GitHub repositories securely to temporary directories
2. **Discovers markdown files** efficiently with prioritization (README first, then docs)
3. **Processes content** using existing smart chunking patterns optimized for markdown
4. **Stores vectors** in Qdrant following the same patterns as `smart_crawl_url`
5. **Manages sources** with GitHub-specific naming: `github.com/{owner}/{repo}`
6. **Cleans up** temporary directories safely after processing

### Success Criteria

- [ ] Tool successfully clones public GitHub repositories
- [ ] Processes all markdown files with smart chunking
- [ ] Stores content in Qdrant with proper metadata and source management
- [ ] Handles errors gracefully (private repos, network issues, large repos)
- [ ] Cleans up temporary directories reliably
- [ ] Integrates seamlessly with existing RAG query tools
- [ ] Follows all existing code patterns and conventions

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- file: E:\mcp-crawl4ai-rag\src\crawl4ai_mcp.py
  why: Main MCP server structure, tool registration patterns, context access patterns

- file: E:\mcp-crawl4ai-rag\src\crawl4ai_mcp.py (smart_crawl_url function)
  why: Exact pattern to follow for content processing, storage, and source management

- file: E:\mcp-crawl4ai-rag\src\qdrant_wrapper.py  
  why: Vector storage patterns, collection schemas, batch processing, source management

- file: E:\mcp-crawl4ai-rag\src\utils\chunking.py
  why: Smart chunking patterns for markdown content processing

- docfile: PRPs/ai_docs/github_cloning_best_practices.md
  why: Security, performance, and implementation best practices for GitHub repository processing

- url: https://docs.github.com/en/rest
  section: Repository API for metadata extraction
  critical: Rate limiting, authentication patterns

- url: https://git-scm.com/docs/git-clone
  section: Clone options and security considerations  
  critical: Shallow clone flags, timeout handling
```

### Current Codebase Tree

```bash
src/
├── crawl4ai_mcp.py          # Main MCP server with smart_crawl_url pattern
├── qdrant_wrapper.py        # Vector storage with source management
├── utils/
│   ├── chunking.py         # Smart markdown chunking
│   ├── embedding.py        # Embedding creation patterns
│   └── validation.py       # URL validation patterns
tests/
├── conftest.py             # Test fixtures and setup
├── test_mcp_server.py      # MCP tool testing patterns
└── test_qdrant_wrapper.py  # Storage testing patterns
```

### Desired Codebase Tree with New Files

```bash
src/
├── crawl4ai_mcp.py          # Modified: Add smart_crawl_github tool
├── qdrant_wrapper.py        # Existing: Use current storage patterns
├── utils/
│   ├── chunking.py         # Existing: Use smart_chunk_markdown
│   ├── embedding.py        # Existing: Use current embedding patterns
│   ├── validation.py       # Modified: Add GitHub URL validation
│   └── github_processor.py # NEW: GitHub repository processing utilities
tests/
├── test_smart_crawl_github.py # NEW: Comprehensive tool tests
└── test_github_processor.py   # NEW: GitHub utility tests
```

### Known Gotchas of Our Codebase & Library Quirks

```python
# CRITICAL: MCP tools must follow exact signature pattern
@mcp.tool()
async def tool_name(ctx: Context, param: str) -> str:
    # Always return JSON string, never dict
    return json.dumps(result, indent=2)

# CRITICAL: Context access pattern for shared resources
qdrant_client = ctx.request_context.lifespan_context.qdrant_client
crawler = ctx.request_context.lifespan_context.crawler

# CRITICAL: Source management - update sources BEFORE adding documents
update_source_info(qdrant_client, source_id, summary, word_count)
add_documents_to_supabase(qdrant_client, urls, chunks, contents, metadatas, url_to_full_document)

# CRITICAL: Smart chunking respects code blocks and paragraphs  
chunks = smart_chunk_markdown(content, chunk_size=chunk_size)

# CRITICAL: Error handling pattern - try/except with JSON error response
try:
    # tool logic
    return json.dumps({"success": True, "data": result}, indent=2)
except Exception as e:
    return json.dumps({"success": False, "error": str(e)}, indent=2)

# CRITICAL: Qdrant point ID generation must be deterministic
point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{chunk_number}"))

# CRITICAL: Virtual URL pattern for repository files
virtual_url = f"https://github.com/{owner}/{repo}/blob/main/{relative_path}#L{chunk_index}"

# GOTCHA: Windows path handling requires proper normalization
relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

# GOTCHA: Git requires specific clone flags for shallow cloning
git clone --depth 1 --single-branch --no-tags [url] [path]

# GOTCHA: Cleanup must be in finally block to handle all exceptions
try:
    # process repository
    pass
finally:
    shutil.rmtree(clone_path, ignore_errors=True)
```

## Implementation Blueprint

### Data Models and Structure

```python
# GitHub repository metadata structure
@dataclass
class GitHubRepoInfo:
    owner: str
    repo_name: str
    clone_url: str
    source_id: str  # "github.com/{owner}/{repo}"
    
# Processed markdown file structure  
@dataclass
class ProcessedMarkdownFile:
    file_path: str          # Relative path from repo root
    content: str           # Processed markdown content
    chunks: List[Dict]     # Smart chunks with metadata
    metadata: Dict         # File-level metadata
```

### List of Tasks to be Completed to Fulfill the PRP

```yaml
Task 1: Create GitHub Processing Utilities
CREATE src/utils/github_processor.py:
  - IMPLEMENT GitHubRepoManager class for secure cloning
  - IMPLEMENT MarkdownDiscovery class for efficient file finding  
  - IMPLEMENT GitHubMetadataExtractor for repository information
  - PATTERN: Follow existing validation.py structure
  - SECURITY: Implement URL validation and resource limits

Task 2: Add GitHub URL Validation
MODIFY src/utils/validation.py:
  - FIND pattern: existing URL validation functions
  - INJECT new validate_github_url function
  - PATTERN: Same return structure as existing validators
  - PRESERVE existing validation logic

Task 3: Create Smart Crawl GitHub Tool
MODIFY src/crawl4ai_mcp.py:
  - FIND pattern: @mcp.tool() decorator usage
  - INJECT smart_crawl_github function after smart_crawl_url
  - MIRROR pattern from smart_crawl_url implementation
  - MODIFY for GitHub-specific processing
  - KEEP identical error handling and response formatting

Task 4: Create Comprehensive Tests
CREATE tests/test_smart_crawl_github.py:
  - MIRROR pattern from tests/test_mcp_server.py
  - IMPLEMENT unit tests for all error scenarios
  - IMPLEMENT integration tests with mock repositories
  - PATTERN: Use existing fixtures from conftest.py

CREATE tests/test_github_processor.py:
  - IMPLEMENT utility function tests
  - PATTERN: Mock-based testing like existing tests
  - COVER security validation and error handling

Task 5: Integration Testing and Validation
MODIFY tests/conftest.py:
  - INJECT GitHub-specific test fixtures  
  - PATTERN: Follow existing fixture patterns
  - ADD mock repository data structures
```

### Task 1 Pseudocode: GitHub Processing Utilities

```python
# src/utils/github_processor.py
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import fnmatch
import logging

class GitHubRepoManager:
    """Secure GitHub repository cloning and management."""
    
    def __init__(self, max_repo_size_mb: int = 500, clone_timeout: int = 300):
        # PATTERN: Initialize with environment-based limits
        self.max_repo_size_mb = max_repo_size_mb
        self.clone_timeout = clone_timeout
        
    async def clone_repository(self, repo_url: str) -> Dict[str, Any]:
        """
        Securely clone repository with validation and resource limits.
        
        SECURITY: Validates URL, blocks private IPs, implements timeouts
        PERFORMANCE: Uses shallow clone for content extraction only
        CLEANUP: Ensures cleanup even on exceptions
        """
        # PATTERN: Validate input first (like existing tools)
        validation = self._validate_github_url(repo_url)
        if not validation["valid"]:
            raise ValueError(validation["error"])
        
        # SECURITY: Create secure temporary directory
        temp_dir = tempfile.mkdtemp(prefix="mcp_github_")
        clone_path = Path(temp_dir)
        
        try:
            # PERFORMANCE: Shallow clone with timeout
            cmd = [
                "git", "clone", 
                "--depth", "1",
                "--single-branch",
                "--no-tags",
                repo_url, str(clone_path)
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.clone_timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
            
            # SECURITY: Check repository size
            repo_size = self._calculate_repo_size(clone_path)
            if repo_size > self.max_repo_size_mb:
                raise ValueError(f"Repository too large: {repo_size}MB")
            
            return {
                "clone_path": clone_path,
                "repo_info": validation,
                "size_mb": repo_size,
                "success": True
            }
            
        except Exception:
            # CRITICAL: Cleanup on any failure
            shutil.rmtree(clone_path, ignore_errors=True)
            raise

class MarkdownDiscovery:
    """Efficient markdown file discovery with prioritization."""
    
    EXCLUDE_PATTERNS = {
        "node_modules/**", ".git/**", "__pycache__/**",
        "venv/**", "build/**", "dist/**", ".cache/**"
    }
    
    MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}
    
    def find_markdown_files(self, repo_path: Path, max_files: int = 500) -> List[Path]:
        """
        Find and prioritize markdown files.
        
        PERFORMANCE: Uses generator pattern, respects file limits
        PRIORITIZATION: README first, then docs/, then others
        """
        # PATTERN: Generator for memory efficiency
        all_files = list(self._discover_files(repo_path, max_files))
        
        # PRIORITIZATION: Sort by importance
        return self._prioritize_files(all_files, repo_path)

class GitHubMetadataExtractor:
    """Extract repository metadata and structure information."""
    
    def extract_repo_metadata(self, repo_path: Path, repo_info: Dict) -> Dict[str, Any]:
        """
        Extract comprehensive repository metadata.
        
        PATTERN: Combine local analysis with API data (if available)
        """
        # Local metadata extraction
        metadata = {
            "readme_files": self._find_readme_files(repo_path),
            "languages": self._detect_languages(repo_path),
            "structure": self._analyze_structure(repo_path)
        }
        
        # Add GitHub-specific info
        metadata.update(repo_info)
        
        return metadata
```

### Task 3 Pseudocode: Smart Crawl GitHub Tool

```python
# src/crawl4ai_mcp.py - Add this function

@mcp.tool()
async def smart_crawl_github(
    ctx: Context,
    repo_url: str,
    max_files: int = 500,
    chunk_size: int = 5000
) -> str:
    """
    Intelligently crawl a GitHub repository and store markdown content.
    
    PATTERN: Mirrors smart_crawl_url structure exactly
    INTEGRATION: Uses existing storage and processing patterns
    SECURITY: Safe cloning with resource limits and cleanup
    """
    try:
        # PATTERN: Get clients from context (like smart_crawl_url)
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        
        # PATTERN: Import utilities (same pattern as other tools)
        from utils.github_processor import GitHubRepoManager, MarkdownDiscovery, GitHubMetadataExtractor
        from utils.chunking import smart_chunk_markdown
        from utils.validation import validate_github_url
        
        # PATTERN: Validate input first
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "error": validation["error"]
            }, indent=2)
        
        # Initialize processors
        repo_manager = GitHubRepoManager()
        markdown_discovery = MarkdownDiscovery()
        metadata_extractor = GitHubMetadataExtractor()
        
        # Clone repository
        clone_result = await repo_manager.clone_repository(repo_url)
        clone_path = clone_result["clone_path"]
        repo_info = clone_result["repo_info"]
        
        try:
            # PATTERN: Source naming like smart_crawl_url
            source_id = f"github.com/{repo_info['owner']}/{repo_info['repo_name']}"
            
            # Discover markdown files  
            markdown_files = markdown_discovery.find_markdown_files(clone_path, max_files)
            
            # Extract repository metadata
            repo_metadata = metadata_extractor.extract_repo_metadata(clone_path, repo_info)
            
            # PATTERN: Process files like smart_crawl_url
            urls = []
            chunk_numbers = []  
            contents = []
            metadatas = []
            url_to_full_document = {}
            
            total_files_processed = 0
            total_chunks = 0
            
            for md_file in markdown_files:
                # Read file content
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # PATTERN: Smart chunking (exact same as smart_crawl_url)
                chunks = smart_chunk_markdown(content, chunk_size)
                
                # Calculate relative path for URL generation
                relative_path = md_file.relative_to(clone_path)
                relative_path_str = str(relative_path).replace("\\", "/")
                
                # PATTERN: Virtual URL generation
                base_virtual_url = f"https://github.com/{repo_info['owner']}/{repo_info['repo_name']}/blob/main/{relative_path_str}"
                url_to_full_document[base_virtual_url] = content
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    virtual_url = f"{base_virtual_url}#L{i}"
                    
                    urls.append(virtual_url)
                    chunk_numbers.append(i)
                    contents.append(chunk)
                    
                    # PATTERN: Metadata structure like smart_crawl_url
                    metadata = {
                        "chunk_index": i,
                        "url": virtual_url,
                        "source": source_id,
                        "file_path": relative_path_str,
                        "repository": f"{repo_info['owner']}/{repo_info['repo_name']}",
                        "content_type": "markdown",
                        "char_count": len(chunk),
                        "word_count": len(chunk.split()),
                        "crawl_type": "github_repository",
                        "crawl_time": datetime.now(timezone.utc).isoformat(),
                        **repo_metadata
                    }
                    metadatas.append(metadata)
                
                total_files_processed += 1
                total_chunks += len(chunks)
            
            # PATTERN: Update source info first (critical order from smart_crawl_url)
            total_word_count = sum(len(content.split()) for content in contents)
            source_summary = f"GitHub repository: {repo_info['owner']}/{repo_info['repo_name']}. {repo_metadata.get('description', 'Repository documentation and markdown files.')}"
            
            update_source_info(qdrant_client, source_id, source_summary, total_word_count)
            
            # PATTERN: Store documents using existing function
            add_documents_to_supabase(
                qdrant_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document
            )
            
            # PATTERN: Success response like smart_crawl_url
            return json.dumps({
                "success": True,
                "repository": f"{repo_info['owner']}/{repo_info['repo_name']}",
                "source_id": source_id,
                "files_processed": total_files_processed,
                "chunks_stored": total_chunks,
                "total_word_count": total_word_count,
                "repository_size_mb": clone_result["size_mb"],
                "repository_metadata": repo_metadata
            }, indent=2)
            
        finally:
            # CRITICAL: Always cleanup
            repo_manager.cleanup_repository(clone_path)
            
    except Exception as e:
        # PATTERN: Error response like other tools
        return json.dumps({
            "success": False,
            "repository_url": repo_url,
            "error": str(e)
        }, indent=2)
```

### Integration Points

```yaml
IMPORTS:
  - add to: src/crawl4ai_mcp.py imports
  - pattern: "from utils.github_processor import GitHubRepoManager"

ENVIRONMENT:
  - add: GITHUB_TOKEN (optional, for private repos)
  - add: MAX_REPO_SIZE_MB (default: 500)
  - add: GIT_CLONE_TIMEOUT (default: 300)

DEPENDENCIES:
  - existing: all current dependencies are sufficient
  - git: requires git command available in PATH
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/utils/github_processor.py --fix
ruff check src/crawl4ai_mcp.py --fix
mypy src/utils/github_processor.py
mypy src/crawl4ai_mcp.py

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests

```python
# CREATE tests/test_smart_crawl_github.py
def test_smart_crawl_github_public_repo(mock_qdrant_client):
    """Test successful processing of public repository"""
    # PATTERN: Use existing test patterns from test_mcp_server.py
    # Mock git clone, file discovery, content processing
    
def test_smart_crawl_github_private_repo_error():
    """Test proper error handling for private repositories"""
    # Should return clear error about authentication
    
def test_smart_crawl_github_invalid_url():
    """Test URL validation error handling"""
    # Should return validation error for malformed URLs
    
def test_smart_crawl_github_large_repo_error():
    """Test repository size limit enforcement"""
    # Should return error for repositories exceeding size limits
    
def test_smart_crawl_github_network_timeout():
    """Test timeout handling for network issues"""
    # Should handle git clone timeouts gracefully

def test_smart_crawl_github_cleanup_on_exception():
    """Test that cleanup happens even when exceptions occur"""
    # Critical test for resource management
```

```bash
# Run and iterate until passing:
uv run pytest tests/test_smart_crawl_github.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Test with a real small public repository
python -c "
import asyncio
import sys
sys.path.append('src')
from crawl4ai_mcp import smart_crawl_github
from unittest.mock import Mock

# Create mock context
ctx = Mock()
ctx.request_context.lifespan_context.qdrant_client = Mock()

# Test with a small public repo
result = asyncio.run(smart_crawl_github(
    ctx, 
    'https://github.com/octocat/Hello-World'
))
print(result)
"

# Expected: JSON response with success=True and repository data
# If error: Check git installation, network connectivity, repo accessibility
```

### Level 4: MCP Server Integration Test

```bash
# Start MCP server and test tool registration
cd src && python -m crawl4ai_mcp

# Test tool is registered and callable
# Verify no startup errors related to new imports
# Check that existing tools still work correctly
```

## Final Validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check src/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Git clone works with public repositories
- [ ] Private repository errors are handled gracefully
- [ ] Large repository size limits are enforced
- [ ] Temporary directories are cleaned up properly
- [ ] Vector storage follows existing patterns exactly
- [ ] Source management integrates with existing tools
- [ ] Error messages are clear and actionable

---

## Anti-Patterns to Avoid

- ❌ Don't create new storage patterns when existing ones work perfectly
- ❌ Don't skip URL validation for "trusted" repositories  
- ❌ Don't ignore cleanup in exception handlers - always use finally blocks
- ❌ Don't hardcode GitHub URLs or repository names in tests
- ❌ Don't use synchronous file I/O in async functions without proper handling
- ❌ Don't process .git directory contents (security and performance risk)
- ❌ Don't modify smart_chunk_markdown function - it works perfectly as-is
- ❌ Don't change existing MCP tool patterns - follow them exactly

## Confidence Score: 9/10

This PRP provides comprehensive context for one-pass implementation success:
- ✅ Complete research of existing patterns to follow
- ✅ Detailed security and performance requirements 
- ✅ Exact code patterns and gotchas from codebase analysis
- ✅ Comprehensive test strategy with existing fixture patterns
- ✅ Executable validation gates with specific commands
- ✅ External documentation for GitHub best practices
- ✅ Clear integration points and dependency requirements

The AI agent has everything needed to implement this feature successfully while maintaining consistency with the existing codebase architecture.