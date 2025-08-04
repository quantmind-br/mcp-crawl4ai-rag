"""
Integration tests for smart_crawl_github multi-file type support.

Tests the complete workflow from repository cloning through multi-file processing
to storage in the vector database.
"""
import pytest
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from crawl4ai_mcp import smart_crawl_github, Context


class TestSmartCrawlGitHubIntegration:
    """Integration tests for smart_crawl_github multi-file functionality."""
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock MCP context."""
        context = Mock(spec=Context)
        return context
    
    @pytest.fixture
    def sample_repo_structure(self):
        """Create a temporary repository with multiple file types."""
        temp_dir = tempfile.mkdtemp(prefix="test_repo_")
        
        # Create directory structure
        os.makedirs(os.path.join(temp_dir, "src"))
        os.makedirs(os.path.join(temp_dir, "docs"))
        os.makedirs(os.path.join(temp_dir, "config"))
        
        # Create README.md
        readme_content = """# Test Repository

This is a test repository for integration testing.

## Features

- Multi-language support
- Configuration management
- Comprehensive documentation

## Getting Started

```bash
npm install
python -m pip install -r requirements.txt
```
"""
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Create Python file with docstrings
        python_content = '''"""
Test Python module for integration testing.

This module demonstrates docstring extraction capabilities.
"""

def calculate_sum(a: int, b: int) -> int:
    """
    Calculate the sum of two integers.
    
    Args:
        a: First integer
        b: Second integer
        
    Returns:
        The sum of a and b
    """
    return a + b

class DataProcessor:
    """
    A class for processing data with various methods.
    
    This class demonstrates class-level documentation.
    """
    
    def __init__(self, name: str):
        """Initialize the processor with a name."""
        self.name = name
    
    async def process_async(self, data: List[str]) -> Dict[str, Any]:
        """
        Process data asynchronously.
        
        Args:
            data: List of strings to process
            
        Returns:
            Processed data as dictionary
        """
        return {"processed": len(data), "name": self.name}
'''
        with open(os.path.join(temp_dir, "src", "processor.py"), "w") as f:
            f.write(python_content)
        
        # Create TypeScript file with JSDoc
        typescript_content = '''
/**
 * User interface definition with comprehensive documentation.
 * @interface User
 * @public
 */
export interface User {
    id: number;
    name: string;
    email: string;
    active: boolean;
}

/**
 * Service class for managing user operations.
 * @class UserService
 * @public
 */
export class UserService {
    private users: User[] = [];
    
    /**
     * Retrieves a user by their ID.
     * @param id - The unique identifier for the user
     * @returns The user object if found, undefined otherwise
     * @example
     * ```typescript
     * const service = new UserService();
     * const user = service.getUser(123);
     * ```
     */
    getUser(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
    
    /**
     * Creates a new user in the system.
     * @param userData - The user data to create
     * @returns Promise resolving to the created user
     * @throws {ValidationError} When user data is invalid
     */
    async createUser(userData: Omit<User, 'id'>): Promise<User> {
        const newUser: User = {
            id: Date.now(),
            ...userData
        };
        this.users.push(newUser);
        return newUser;
    }
}

/**
 * Utility function for validating email addresses.
 * @param email - Email address to validate
 * @returns True if email is valid, false otherwise
 */
export function validateEmail(email: string): boolean {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}
'''
        with open(os.path.join(temp_dir, "src", "user-service.ts"), "w") as f:
            f.write(typescript_content)
        
        # Create package.json
        package_json = {
            "name": "test-integration-repo",
            "version": "1.0.0",
            "description": "Test repository for integration testing",
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "test": "jest",
                "build": "tsc"
            },
            "dependencies": {
                "express": "^4.18.0",
                "typescript": "^4.9.0"
            },
            "devDependencies": {
                "jest": "^29.0.0",
                "@types/node": "^18.0.0"
            }
        }
        with open(os.path.join(temp_dir, "package.json"), "w") as f:
            json.dump(package_json, f, indent=2)
        
        # Create docker-compose.yml
        docker_compose = """version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - PORT=3000
    volumes:
      - ./src:/app/src
    depends_on:
      - database
  
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""
        with open(os.path.join(temp_dir, "config", "docker-compose.yml"), "w") as f:
            f.write(docker_compose)
        
        # Create pyproject.toml
        pyproject_content = """[project]
name = "test-python-project"
version = "0.1.0"
description = "Test Python project for integration testing"
authors = [
    {name = "Test Author", email = "test@example.com"}
]
readme = "README.md"
license = {text = "MIT"}

[project.dependencies]
requests = "^2.28.0"
click = "^8.1.0"
pydantic = "^1.10.0"

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "mypy>=0.991"
]

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
"""
        with open(os.path.join(temp_dir, "pyproject.toml"), "w") as f:
            f.write(pyproject_content)
        
        # Create API documentation
        api_docs = """# API Documentation

## User Management API

### Endpoints

#### GET /users/{id}
Retrieve a user by ID.

**Parameters:**
- `id` (integer): User ID

**Response:**
```json
{
  "id": 123,
  "name": "John Doe",
  "email": "john@example.com",
  "active": true
}
```

#### POST /users
Create a new user.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "active": true
}
```
"""
        with open(os.path.join(temp_dir, "docs", "api.md"), "w") as f:
            f.write(api_docs)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_multi_file_complete_workflow(
        self, 
        mock_extractor_class,
        mock_manager_class, 
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test complete workflow with multiple file types."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "owner": "test",
            "repo_name": "repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test with multiple file types
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo",
            max_files=50,
            chunk_size=1000,
            max_size_mb=500,
            file_types_to_index=['.md', '.py', '.ts', '.json', '.yml', '.toml']
        )
        
        # Parse result
        result_data = json.loads(result)
        
        # Verify success
        assert result_data["status"] == "success"
        assert result_data["repo_url"] == "https://github.com/test/repo"
        
        # Verify file type processing
        assert "file_types_processed" in result_data
        file_types = result_data["file_types_processed"]
        
        # Should have processed multiple types
        assert file_types["markdown"] > 0  # README.md, api.md
        assert file_types["python"] > 0   # processor.py
        assert file_types["typescript"] > 0  # user-service.ts
        assert file_types["configuration"] > 0  # package.json, docker-compose.yml, pyproject.toml
        
        # Verify chunks were created
        assert result_data["total_chunks"] > 0
        assert result_data["total_files_processed"] >= 6  # Multiple files processed
        
        # Verify documents were added to storage
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        # Check document types and metadata
        doc_types = set()
        languages = set()
        for doc in added_documents:
            metadata = doc["metadata"]
            doc_types.add(metadata["type"])
            languages.add(metadata["language"])
        
        # Should have various document types
        expected_types = {"markdown", "module", "function", "class", "interface", "configuration"}
        assert len(doc_types.intersection(expected_types)) >= 4
        
        # Should have various languages
        expected_languages = {"markdown", "python", "typescript", "json", "yaml", "toml"}
        assert len(languages.intersection(expected_languages)) >= 4
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_backward_compatibility_markdown_only(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test backward compatibility with default markdown-only behavior."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test with default parameters (should only process markdown)
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo"
            # No file_types_to_index parameter - should default to ['.md']
        )
        
        # Parse result
        result_data = json.loads(result)
        
        # Verify success
        assert result_data["status"] == "success"
        
        # Verify only markdown was processed
        file_types = result_data["file_types_processed"]
        assert file_types["markdown"] > 0  # Should have markdown files
        assert file_types.get("python", 0) == 0  # Should not have Python
        assert file_types.get("typescript", 0) == 0  # Should not have TypeScript
        assert file_types.get("configuration", 0) == 0  # Should not have config
        
        # Verify documents were added
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        # All documents should be markdown
        for doc in added_documents:
            assert doc["metadata"]["language"] == "markdown"
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_single_file_type_processing(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test processing single specific file type."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test with only Python files
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo",
            file_types_to_index=['.py']
        )
        
        result_data = json.loads(result)
        
        # Should only have Python processing
        file_types = result_data["file_types_processed"]
        assert file_types["python"] > 0
        assert file_types.get("markdown", 0) == 0
        assert file_types.get("typescript", 0) == 0
        
        # Verify Python-specific content
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        python_docs = [doc for doc in added_documents if doc["metadata"]["language"] == "python"]
        assert len(python_docs) > 0
        
        # Should have various Python elements
        doc_types = {doc["metadata"]["type"] for doc in python_docs}
        assert "module" in doc_types  # Module docstring
        assert "function" in doc_types  # Function docstrings
        assert "class" in doc_types  # Class docstrings
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_error_handling_invalid_repo(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_context
    ):
        """Test error handling with invalid repository."""
        # Setup mock to raise error
        mock_manager = Mock()
        mock_manager.clone_repository.side_effect = ValueError("Invalid GitHub repository URL")
        mock_manager_class.return_value = mock_manager
        
        # Test with invalid repo URL
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://invalid-url.com/not/github",
            file_types_to_index=['.md', '.py']
        )
        
        result_data = json.loads(result)
        
        # Should return error status
        assert result_data["status"] == "error"
        assert "Invalid GitHub repository URL" in result_data["error"]
        assert result_data["total_chunks"] == 0
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_mixed_file_processing_with_errors(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test processing with mixed valid and problematic files."""
        # Create problematic files in the sample repo
        
        # Add invalid Python file (syntax error)
        invalid_python = '''
def broken_function(
    # Missing closing parenthesis and invalid syntax
    return "this will fail"
'''
        with open(os.path.join(sample_repo_structure, "src", "broken.py"), "w") as f:
            f.write(invalid_python)
        
        # Add very large file (should be skipped)
        large_content = "x" * 2_000_000  # 2MB content
        with open(os.path.join(sample_repo_structure, "src", "large.py"), "w") as f:
            f.write(large_content)
        
        # Add binary-like file
        with open(os.path.join(sample_repo_structure, "binary.json"), "wb") as f:
            f.write(b"{\x00binary\x00content\x00}")
        
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test processing - should continue despite errors
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo",
            file_types_to_index=['.md', '.py', '.ts', '.json']
        )
        
        result_data = json.loads(result)
        
        # Should still succeed overall
        assert result_data["status"] == "success"
        
        # Should have processed valid files
        assert result_data["total_chunks"] > 0
        file_types = result_data["file_types_processed"]
        assert file_types["markdown"] > 0  # Valid markdown files
        assert file_types["python"] > 0   # Valid Python files (excluding broken ones)
        assert file_types["typescript"] > 0  # Valid TypeScript files
        
        # Documents should be added (only valid ones)
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        # Should not contain content from problematic files
        all_content = " ".join(doc["content"] for doc in added_documents)
        assert "broken_function" not in all_content  # Syntax error file skipped
        assert len(all_content) < 1_500_000  # Large file not included
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_metadata_structure_validation(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test that generated metadata follows expected structure."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "owner": "test",
            "repo_name": "repo",
            "full_name": "test/repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test processing
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo",
            file_types_to_index=['.md', '.py', '.ts', '.json', '.yml', '.toml']
        )
        
        result_data = json.loads(result)
        assert result_data["status"] == "success"
        
        # Verify documents were added with correct metadata structure
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        # Validate metadata structure for each document
        required_fields = [
            "file_path", "type", "name", "language", "repo_url", "source_type"
        ]
        
        for doc in added_documents:
            assert "content" in doc
            assert "metadata" in doc
            
            metadata = doc["metadata"]
            
            # Check required fields
            for field in required_fields:
                assert field in metadata, f"Missing field {field} in metadata"
            
            # Validate field types and values
            assert isinstance(metadata["file_path"], str)
            assert isinstance(metadata["type"], str)
            assert isinstance(metadata["name"], str)
            assert isinstance(metadata["language"], str)
            assert metadata["repo_url"] == "https://github.com/test/repo"
            assert metadata["source_type"] == "github_repository"
            
            # Language-specific validation
            if metadata["language"] == "python":
                assert metadata["type"] in ["module", "function", "class"]
                if metadata["type"] == "function":
                    assert "signature" in metadata
                    assert "line_number" in metadata
            
            elif metadata["language"] == "typescript":
                assert metadata["type"] in ["function", "class", "interface", "method"]
                if metadata["type"] in ["function", "method"]:
                    assert "signature" in metadata
            
            elif metadata["language"] == "markdown":
                assert metadata["type"] == "markdown"
            
            elif metadata["language"] in ["json", "yaml", "toml"]:
                assert metadata["type"] == "configuration"
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.add_documents_to_supabase')
    @patch('crawl4ai_mcp.GitHubRepoManager')
    @patch('crawl4ai_mcp.GitHubMetadataExtractor')
    async def test_chunking_integration(
        self,
        mock_extractor_class,
        mock_manager_class,
        mock_add_documents,
        mock_context,
        sample_repo_structure
    ):
        """Test that chunking works correctly with multi-file content."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.clone_repository.return_value = sample_repo_structure
        mock_manager_class.return_value = mock_manager
        
        mock_extractor = Mock()
        mock_extractor.extract_repo_metadata.return_value = {
            "repo_url": "https://github.com/test/repo",
            "source_type": "github_repository"
        }
        mock_extractor_class.return_value = mock_extractor
        
        mock_add_documents.return_value = None
        
        # Test with small chunk size to force chunking
        result = await smart_crawl_github(
            ctx=mock_context,
            repo_url="https://github.com/test/repo",
            chunk_size=500,  # Small chunks
            file_types_to_index=['.md', '.py', '.ts']
        )
        
        result_data = json.loads(result)
        assert result_data["status"] == "success"
        
        # Verify chunking occurred
        mock_add_documents.assert_called_once()
        added_documents = mock_add_documents.call_args[0][0]
        
        # Should have multiple chunks due to small chunk size
        assert len(added_documents) > 3
        
        # Verify chunk metadata
        chunk_counts = {}
        for doc in added_documents:
            metadata = doc["metadata"]
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            
            # Track chunks per document type
            doc_key = f"{metadata['file_path']}:{metadata['name']}"
            if doc_key not in chunk_counts:
                chunk_counts[doc_key] = 0
            chunk_counts[doc_key] += 1
        
        # Some documents should have been chunked
        multi_chunk_docs = [count for count in chunk_counts.values() if count > 1]
        assert len(multi_chunk_docs) > 0  # At least some documents were chunked
        
        # Verify chunk ordering
        for doc in added_documents:
            metadata = doc["metadata"]
            assert metadata["chunk_index"] >= 0
            assert metadata["chunk_index"] < metadata["total_chunks"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])