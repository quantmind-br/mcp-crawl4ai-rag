# GitHub Repository Cloning Best Practices

This document provides critical best practices for securely cloning and processing GitHub repositories in production systems.

## Security Considerations

### URL Validation
- **Always validate GitHub URLs** before cloning to prevent SSRF attacks
- **Block private IP ranges**: localhost, 127.0.0.1, 192.168.x.x, 10.x.x.x, 172.16-31.x.x
- **Use regex patterns** to validate GitHub URL format: `https://github\.com/[\w\-\.]+/[\w\-\.]+(?:\.git)?/?$`

### Safe Cloning Commands
```bash
# Shallow clone for content extraction (faster, less bandwidth)
git clone --depth 1 --single-branch --no-tags [repo_url] [local_path]

# Clone specific branch if needed
git clone --depth 1 --branch main --single-branch [repo_url] [local_path]

# Clone with timeout (prevent hanging)
timeout 300 git clone --depth 1 [repo_url] [local_path]
```

### Resource Management
- **Set size limits**: Check repository size before full clone
- **Use temporary directories**: Always use secure temp directories with proper cleanup
- **Implement timeouts**: 5 minutes for cloning, prevent hanging operations
- **Limit concurrent clones**: Use semaphores to prevent resource exhaustion

## Authentication Patterns

### Token-Based Authentication
```python
# For private repositories
authenticated_url = f"https://{github_token}@github.com/{owner}/{repo}.git"

# Environment variable for token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
```

### Error Handling for Authentication
- **Private repositories**: Return clear error messages about authentication needs
- **Rate limiting**: Handle GitHub API rate limits gracefully
- **Invalid credentials**: Distinguish between auth failures and other errors

## Efficient File Discovery

### Markdown File Patterns
```python
# Common markdown extensions
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}

# Priority patterns for important files
README_PATTERNS = ["README.md", "readme.md", "Readme.md", "README.rst", "README.txt"]
CHANGELOG_PATTERNS = ["CHANGELOG.md", "HISTORY.md", "NEWS.md", "RELEASES.md"]
```

### Exclusion Patterns
```python
# Standard directories to skip (performance and security)
EXCLUDE_PATTERNS = {
    "node_modules/**",    # npm packages
    ".git/**",           # git metadata
    ".vscode/**",        # editor configs
    "__pycache__/**",    # Python cache
    "venv/**",           # Python virtual env
    "build/**",          # Build artifacts
    "dist/**",           # Distribution files
    ".cache/**",         # Cache directories
    "coverage/**"        # Test coverage
}
```

## Repository Metadata Extraction

### Local Git Information
```bash
# Get repository info from local clone
git remote get-url origin          # Repository URL
git branch --show-current          # Current branch
git log -1 --format="%H %s %an"   # Latest commit info
```

### GitHub API Integration
```python
# Repository metadata via API (if token available)
GET https://api.github.com/repos/{owner}/{repo}
# Returns: description, topics, language, stars, license, etc.

# Language statistics
GET https://api.github.com/repos/{owner}/{repo}/languages
# Returns: language breakdown by bytes
```

## Performance Optimization

### Memory Management
- **Stream processing**: Process files one at a time for large repositories
- **Generator patterns**: Use generators for file discovery to avoid loading all paths
- **Chunk processing**: Process content in chunks, not entire files at once

### Bandwidth Optimization
- **Shallow clones**: Use `--depth 1` for content extraction only
- **Sparse checkout**: For very large repos, consider sparse-checkout patterns
- **Compression**: Git automatically compresses, but be aware of network costs

## Error Recovery Patterns

### Common Error Scenarios
1. **Repository not found**: Invalid URL or private repository
2. **Authentication required**: Private repository without token
3. **Network timeouts**: Large repositories or slow connections
4. **Size limits exceeded**: Repository too large for processing
5. **Permission denied**: Insufficient access rights

### Graceful Degradation
```python
try:
    result = clone_repository(repo_url)
except AuthenticationError:
    return {"error": "private_repository", "suggestion": "provide_github_token"}
except TimeoutError:
    return {"error": "timeout", "suggestion": "repository_too_large_or_slow_network"}
except NetworkError:
    return {"error": "network", "suggestion": "check_connectivity_and_retry"}
```

## Cleanup Best Practices

### Secure Cleanup
```python
import shutil
import tempfile

# Create secure temporary directory
temp_dir = tempfile.mkdtemp(prefix="repo_clone_", dir="/secure/temp")

try:
    # Process repository
    pass
finally:
    # Always cleanup, even on exceptions
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### Cleanup Verification
- **Check disk space**: Verify cleanup actually freed space
- **Monitor temp directories**: Periodic cleanup of orphaned directories
- **Log cleanup results**: Track successful and failed cleanups

## Integration with MCP Patterns

### Source Naming Convention
```python
# For GitHub repositories
source_id = f"github.com/{owner}/{repo}"

# For specific content types
wiki_source = f"github.com/{owner}/{repo}/wiki"
releases_source = f"github.com/{owner}/{repo}/releases"
```

### Content Chunking for Git
- **Respect file boundaries**: Don't split across different files unnecessarily
- **Preserve section headers**: Maintain markdown structure in chunks
- **Include file context**: Add file path and section information to metadata

## Common Pitfalls

### ❌ What NOT to Do
- Don't clone without size checks (can fill disk)
- Don't use force flags (`git clone --force`) 
- Don't ignore authentication errors silently
- Don't process .git directory contents (security risk)
- Don't hardcode GitHub URLs or tokens
- Don't forget cleanup on exceptions

### ✅ Best Practices
- Always validate URLs before cloning
- Use shallow clones for content extraction
- Implement proper resource limits and timeouts
- Clean up temporary directories in finally blocks
- Handle authentication gracefully with clear error messages
- Log operations for debugging but not sensitive information

## External References

- GitHub API Documentation: https://docs.github.com/en/rest
- Git Clone Documentation: https://git-scm.com/docs/git-clone
- GitHub Security Best Practices: https://docs.github.com/en/code-security
- Python tempfile module: https://docs.python.org/3/library/tempfile.html