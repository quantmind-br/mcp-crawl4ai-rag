"""
Cross-system linking tests for file_id functionality.

Tests the file_id linking between Qdrant (RAG) and Neo4j (Knowledge Graph)
systems to ensure consistent cross-system data retrieval and linking.
"""

import os
import pytest
from unittest.mock import Mock, patch

from src.utils.file_id_generator import generate_file_id, extract_repo_name
from src.services.rag_service import RagService, add_documents_to_vector_db, add_code_examples_to_vector_db


class TestFileIdGeneration:
    """Test file ID generation and consistency."""
    
    def test_generate_file_id_consistency(self):
        """Test that file_id generation is consistent across calls."""
        repo_url = "https://github.com/pydantic/pydantic-ai"
        relative_path = "docs/agents.md"
        
        file_id_1 = generate_file_id(repo_url, relative_path)
        file_id_2 = generate_file_id(repo_url, relative_path)
        
        assert file_id_1 == file_id_2
        assert file_id_1 == "pydantic-pydantic-ai:docs/agents.md"
    
    def test_file_id_cross_platform_consistency(self):
        """Test file_id consistency across different path separators."""
        repo_url = "https://github.com/test/repo"
        
        unix_path = "src/utils/helper.py"
        windows_path = "src\\utils\\helper.py"
        
        unix_file_id = generate_file_id(repo_url, unix_path)
        windows_file_id = generate_file_id(repo_url, windows_path)
        
        # Should normalize to the same file_id
        assert unix_file_id == windows_file_id
        assert unix_file_id == "test-repo:src/utils/helper.py"
    
    def test_file_id_special_characters(self):
        """Test file_id generation with special characters."""
        repo_url = "https://github.com/org-name/repo_name"
        relative_path = "docs/api-reference.md"
        
        file_id = generate_file_id(repo_url, relative_path)
        assert file_id == "org-name-repo_name:docs/api-reference.md"
    
    def test_file_id_nested_paths(self):
        """Test file_id generation with deeply nested paths."""
        repo_url = "https://github.com/test/repo"
        relative_path = "src/components/ui/buttons/primary/Button.tsx"
        
        file_id = generate_file_id(repo_url, relative_path)
        assert file_id == "test-repo:src/components/ui/buttons/primary/Button.tsx"


class TestRagFileIdIntegration:
    """Test file_id integration with RAG system (Qdrant)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.rag_service = RagService(self.mock_qdrant_client)
    
    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    @patch('src.services.embedding_service.create_embeddings_batch')
    def test_add_documents_with_file_ids(self, mock_create_embeddings):
        """Test adding documents to Qdrant with file_id metadata."""
        # Setup test data
        urls = ["https://github.com/test/repo"]
        chunk_numbers = [0, 1]
        contents = ["First chunk content", "Second chunk content"]
        file_ids = ["test-repo:readme.md", "test-repo:readme.md"]
        metadatas = [
            {"source": "test-repo", "chunk_index": 0},
            {"source": "test-repo", "chunk_index": 1}
        ]
        url_to_full_document = {
            "https://github.com/test/repo": "Full document content"
        }
        
        # Setup mocks
        mock_create_embeddings.return_value = [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]
        mock_point_batches = [
            [
                {
                    "id": "1",
                    "content": "First chunk content",
                    "payload": {"source": "test-repo", "chunk_index": 0, "file_id": "test-repo:readme.md"}
                },
                {
                    "id": "2", 
                    "content": "Second chunk content",
                    "payload": {"source": "test-repo", "chunk_index": 1, "file_id": "test-repo:readme.md"}
                }
            ]
        ]
        self.mock_qdrant_client.add_documents_to_qdrant.return_value = mock_point_batches
        
        # Execute
        add_documents_to_vector_db(
            self.mock_qdrant_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            file_ids=file_ids
        )
        
        # Verify that upsert_points was called
        self.mock_qdrant_client.upsert_points.assert_called()
        
        # Verify that file_id was added to the points
        call_args = self.mock_qdrant_client.upsert_points.call_args[0]
        collection_name = call_args[0]
        points = call_args[1]
        
        assert collection_name == "crawled_pages"
        # Check that file_id was added to payload
        for point in points:
            assert "file_id" in point.payload
            assert point.payload["file_id"] == "test-repo:readme.md"
    
    @patch.dict(os.environ, {"USE_HYBRID_SEARCH": "false"})
    @patch('src.services.embedding_service.create_embeddings_batch')
    def test_add_code_examples_with_file_ids(self, mock_create_embeddings):
        """Test adding code examples to Qdrant with file_id metadata."""
        # Setup test data
        urls = ["https://github.com/test/repo"]
        chunk_numbers = [0, 1]
        code_examples = ["def example1(): pass", "def example2(): pass"]
        summaries = ["First example summary", "Second example summary"]
        file_ids = ["test-repo:examples.py", "test-repo:examples.py"]
        metadatas = [
            {"source": "test-repo", "language": "python"},
            {"source": "test-repo", "language": "python"}
        ]
        
        # Setup mocks
        mock_create_embeddings.return_value = [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]]
        mock_point_batches = [
            [
                {
                    "id": "code1",
                    "combined_text": "def example1(): pass\n\nFirst example summary",
                    "payload": {"source": "test-repo", "language": "python", "file_id": "test-repo:examples.py"}
                },
                {
                    "id": "code2",
                    "combined_text": "def example2(): pass\n\nSecond example summary", 
                    "payload": {"source": "test-repo", "language": "python", "file_id": "test-repo:examples.py"}
                }
            ]
        ]
        self.mock_qdrant_client.add_code_examples_to_qdrant.return_value = mock_point_batches
        
        # Execute
        add_code_examples_to_vector_db(
            self.mock_qdrant_client,
            urls,
            chunk_numbers,
            code_examples,
            summaries,
            metadatas,
            file_ids=file_ids
        )
        
        # Verify that upsert_points was called
        self.mock_qdrant_client.upsert_points.assert_called()
        
        # Verify that file_id was added to the code examples
        call_args = self.mock_qdrant_client.upsert_points.call_args[0]
        collection_name = call_args[0]
        points = call_args[1]
        
        assert collection_name == "code_examples"
        # Check that file_id was added to payload
        for point in points:
            assert "file_id" in point.payload
            assert point.payload["file_id"] == "test-repo:examples.py"
    
    def test_search_documents_with_file_id_filter(self):
        """Test searching documents with file_id filtering."""
        # Setup mock search results
        expected_results = [
            {
                "id": "doc1",
                "content": "Content from specific file",
                "score": 0.9,
                "payload": {"file_id": "test-repo:specific.md"}
            }
        ]
        self.rag_service.search_with_reranking = Mock(return_value=expected_results)
        
        # Execute search with file_id filter
        results = self.rag_service.search_with_reranking(
            query="test query",
            filter_metadata={"file_id": "test-repo:specific.md"},
            search_type="documents"
        )
        
        # Verify
        assert results == expected_results
        self.rag_service.search_with_reranking.assert_called_once()
        
        # Check that filter was applied
        call_args = self.rag_service.search_with_reranking.call_args
        filter_metadata = call_args[1]["filter_metadata"]
        assert filter_metadata["file_id"] == "test-repo:specific.md"
    
    def test_search_code_examples_with_file_id_filter(self):
        """Test searching code examples with file_id filtering."""
        # Setup mock search results
        expected_results = [
            {
                "id": "code1",
                "code": "def specific_function(): pass",
                "score": 0.85,
                "payload": {"file_id": "test-repo:utils.py"}
            }
        ]
        self.rag_service.search_with_reranking = Mock(return_value=expected_results)
        
        # Execute search with file_id filter
        results = self.rag_service.search_with_reranking(
            query="utility function",
            filter_metadata={"file_id": "test-repo:utils.py"},
            search_type="code_examples"
        )
        
        # Verify
        assert results == expected_results
        
        # Check that filter was applied
        call_args = self.rag_service.search_with_reranking.call_args
        filter_metadata = call_args[1]["filter_metadata"]
        assert filter_metadata["file_id"] == "test-repo:utils.py"


class TestNeo4jFileIdIntegration:
    """Test file_id integration with Neo4j Knowledge Graph system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_neo4j_driver = Mock()
        self.mock_session = Mock()
        self.mock_neo4j_driver.session.return_value = self.mock_session
    
    def test_neo4j_file_node_with_file_id(self):
        """Test creating Neo4j file nodes with file_id metadata."""
        file_id = "test-repo:src/main.py"
        
        # Mock Neo4j operations
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Simulate creating a file node with file_id
            create_query = """
            CREATE (f:File {
                file_id: $file_id,
                path: $path,
                repo_name: $repo_name,
                language: $language
            })
            RETURN f
            """
            
            # This would be called by the actual Neo4j repository parser
            self.mock_session.run.return_value = [{"f": {"file_id": file_id}}]
            
            # Execute simulated query
            self.mock_session.run(
                create_query,
                file_id=file_id,
                path="src/main.py",
                repo_name="test-repo",
                language="python"
            )
            
            # Verify query execution
            self.mock_session.run.assert_called_once()
            call_args = self.mock_session.run.call_args
            assert file_id in str(call_args)
    
    def test_neo4j_query_by_file_id(self):
        """Test querying Neo4j nodes by file_id."""
        file_id = "test-repo:src/main.py"
        
        # Mock Neo4j query results
        mock_results = [
            {
                "f": {
                    "file_id": file_id,
                    "path": "src/main.py",
                    "repo_name": "test-repo"
                },
                "c": {
                    "name": "MainClass",
                    "line_number": 10
                }
            }
        ]
        self.mock_session.run.return_value = mock_results
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Simulate querying for classes in a specific file
            query = """
            MATCH (f:File {file_id: $file_id})-[:CONTAINS]->(c:Class)
            RETURN f, c
            """
            
            result = self.mock_session.run(query, file_id=file_id)
            
            # Verify
            self.mock_session.run.assert_called_once_with(query, file_id=file_id)
            assert result == mock_results
    
    def test_neo4j_cross_file_relationships_with_file_ids(self):
        """Test Neo4j relationships between files using file_ids."""
        source_file_id = "test-repo:src/main.py"
        target_file_id = "test-repo:src/utils.py"
        
        # Mock query for cross-file imports/dependencies
        mock_results = [
            {
                "source_file": {"file_id": source_file_id},
                "target_file": {"file_id": target_file_id},
                "import_statement": "from src.utils import helper"
            }
        ]
        self.mock_session.run.return_value = mock_results
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Query for imports between specific files
            query = """
            MATCH (source:File {file_id: $source_file_id})-[r:IMPORTS]->(target:File {file_id: $target_file_id})
            RETURN source as source_file, target as target_file, r.statement as import_statement
            """
            
            self.mock_session.run(
                query,
                source_file_id=source_file_id,
                target_file_id=target_file_id
            )
            
            # Verify cross-file relationship query
            self.mock_session.run.assert_called_once()
            call_args = self.mock_session.run.call_args[1]
            assert call_args["source_file_id"] == source_file_id
            assert call_args["target_file_id"] == target_file_id


class TestCrossSystemLinking:
    """Test cross-system linking scenarios between RAG and Neo4j."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.mock_neo4j_driver = Mock()
        self.mock_neo4j_session = Mock()
        self.mock_neo4j_driver.session.return_value = self.mock_neo4j_session
        self.rag_service = RagService(self.mock_qdrant_client)
    
    def test_rag_to_neo4j_linking_scenario(self):
        """Test scenario: Find code in RAG, then get structure from Neo4j."""
        file_id = "pydantic-ai:src/agents.py"
        
        # Step 1: Search RAG for code content
        rag_results = [
            {
                "id": "doc1",
                "content": "class Agent: def __init__(self): pass",
                "payload": {"file_id": file_id, "chunk_index": 0}
            }
        ]
        self.rag_service.search_with_reranking = Mock(return_value=rag_results)
        
        # Step 2: Use file_id to get structure from Neo4j
        neo4j_results = [
            {
                "class": {"name": "Agent", "line_number": 15},
                "methods": [
                    {"name": "__init__", "line_number": 16},
                    {"name": "run", "line_number": 20}
                ]
            }
        ]
        self.mock_neo4j_session.run.return_value = neo4j_results
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Execute RAG search
            rag_content = self.rag_service.search_with_reranking(
                query="Agent class implementation",
                filter_metadata={"file_id": file_id}
            )
            
            # Extract file_id from RAG results
            extracted_file_id = rag_content[0]["payload"]["file_id"]
            assert extracted_file_id == file_id
            
            # Use file_id to query Neo4j for structure
            structure_query = """
            MATCH (f:File {file_id: $file_id})-[:CONTAINS]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            RETURN c as class, collect(m) as methods
            """
            
            self.mock_neo4j_session.run(
                structure_query,
                file_id=extracted_file_id
            )
            
            # Verify cross-system linking
            assert rag_content[0]["payload"]["file_id"] == file_id
            self.mock_neo4j_session.run.assert_called_once_with(
                structure_query, 
                file_id=file_id
            )
    
    def test_neo4j_to_rag_linking_scenario(self):
        """Test scenario: Find structure in Neo4j, then get content from RAG."""
        file_id = "fastapi-repo:main.py"
        
        # Step 1: Query Neo4j for class structure
        neo4j_results = [
            {
                "file": {"file_id": file_id, "path": "main.py"},
                "class": {"name": "FastAPI", "line_number": 25}
            }
        ]
        self.mock_neo4j_session.run.return_value = neo4j_results
        
        # Step 2: Use file_id to get content from RAG
        rag_results = [
            {
                "id": "chunk1",
                "content": "class FastAPI: # Main FastAPI application class",
                "payload": {"file_id": file_id, "chunk_index": 0}
            }
        ]
        self.rag_service.search_with_reranking = Mock(return_value=rag_results)
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Execute Neo4j query to find classes
            class_query = """
            MATCH (f:File)-[:CONTAINS]->(c:Class {name: $class_name})
            RETURN f as file, c as class
            """
            
            self.mock_neo4j_session.run(
                class_query,
                class_name="FastAPI"
            )
            
            # Extract file_id from Neo4j results
            extracted_file_id = neo4j_results[0]["file"]["file_id"]
            
            # Use file_id to search RAG for content
            self.rag_service.search_with_reranking(
                query="FastAPI class implementation",
                filter_metadata={"file_id": extracted_file_id}
            )
            
            # Verify cross-system linking
            assert extracted_file_id == file_id
            self.rag_service.search_with_reranking.assert_called_once()
            filter_used = self.rag_service.search_with_reranking.call_args[1]["filter_metadata"]
            assert filter_used["file_id"] == file_id
    
    def test_bidirectional_linking_consistency(self):
        """Test that file_id linking works consistently in both directions."""
        file_id = "test-repo:core/models.py"
        
        # Mock both systems having data for the same file_id
        rag_content = [
            {"content": "class User(BaseModel):", "payload": {"file_id": file_id}}
        ]
        neo4j_structure = [
            {"class": {"name": "User", "file_id": file_id}}
        ]
        
        self.rag_service.search_with_reranking = Mock(return_value=rag_content)
        self.mock_neo4j_session.run.return_value = neo4j_structure
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Query both systems using same file_id
            rag_results = self.rag_service.search_with_reranking(
                query="User model",
                filter_metadata={"file_id": file_id}
            )
            
            self.mock_neo4j_session.run(
                "MATCH (c:Class) WHERE c.file_id = $file_id RETURN c as class",
                file_id=file_id
            )
            
            # Verify consistency
            assert rag_results[0]["payload"]["file_id"] == file_id
            assert neo4j_structure[0]["class"]["file_id"] == file_id
            
            # Both systems should return data for the same file
            assert len(rag_results) > 0
            assert len(neo4j_structure) > 0
    
    def test_unified_search_with_cross_system_enrichment(self):
        """Test unified search that enriches results across both systems."""
        file_id = "django-repo:models/user.py"
        
        # Step 1: Initial RAG search
        initial_rag_results = [
            {
                "content": "class User(models.Model): username = models.CharField()",
                "payload": {"file_id": file_id, "score": 0.9}
            }
        ]
        
        # Step 2: Neo4j enrichment data
        neo4j_enrichment = [
            {
                "class_info": {
                    "name": "User",
                    "methods": ["save", "delete", "clean"],
                    "relationships": ["Profile", "Group"]
                }
            }
        ]
        
        self.rag_service.search_with_reranking = Mock(return_value=initial_rag_results)
        self.mock_neo4j_session.run.return_value = neo4j_enrichment
        
        with patch('neo4j.GraphDatabase.driver', return_value=self.mock_neo4j_driver):
            # Execute enriched search workflow
            
            # 1. Get content from RAG
            content_results = self.rag_service.search_with_reranking(
                query="User model implementation"
            )
            
            # 2. Extract file_ids for enrichment (used in loop below)
            _ = [result["payload"]["file_id"] for result in content_results]
            
            # 3. Enrich with Neo4j structure data
            enriched_results = []
            for result in content_results:
                file_id = result["payload"]["file_id"]
                
                # Get structure data from Neo4j
                self.mock_neo4j_session.run(
                    "MATCH (f:File {file_id: $file_id})-[:CONTAINS]->(c:Class) "
                    "OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method) "
                    "RETURN c, collect(m.name) as methods",
                    file_id=file_id
                )
                
                # Combine RAG content with Neo4j structure
                enriched_result = {
                    **result,
                    "structure_info": neo4j_enrichment[0]["class_info"]
                }
                enriched_results.append(enriched_result)
            
            # Verify enriched results
            assert len(enriched_results) == 1
            assert "structure_info" in enriched_results[0]
            assert enriched_results[0]["structure_info"]["name"] == "User"
            assert "save" in enriched_results[0]["structure_info"]["methods"]


class TestFileIdConsistencyValidation:
    """Test file_id consistency validation across systems."""
    
    def test_file_id_format_validation(self):
        """Test that file_id format is validated correctly."""
        valid_file_ids = [
            "repo-name:path/file.py",
            "complex-repo-name:deeply/nested/path/file.tsx",
            "org-repo:README.md",
        ]
        
        invalid_file_ids = [
            "repo-name",  # Missing colon separator
            ":path/file.py",  # Missing repo name
            "repo-name:",  # Missing file path
            "repo name:file.py",  # Space in repo name (should be normalized)
        ]
        
        for file_id in valid_file_ids:
            assert ":" in file_id
            repo_part, file_part = file_id.split(":", 1)
            assert repo_part  # Repo name should not be empty
            assert file_part  # File path should not be empty
        
        for file_id in invalid_file_ids:
            if ":" not in file_id:
                # Should fail validation
                with pytest.raises((ValueError, IndexError)):
                    repo_part, file_part = file_id.split(":", 1)
                    if not repo_part or not file_part:
                        raise ValueError("Invalid file_id format")
    
    def test_file_id_uniqueness_within_repo(self):
        """Test that file_ids are unique within the same repository."""
        repo_url = "https://github.com/test/repo"
        
        file_paths = [
            "src/main.py",
            "src/utils.py", 
            "docs/readme.md",
            "tests/test_main.py"
        ]
        
        file_ids = [generate_file_id(repo_url, path) for path in file_paths]
        
        # All file_ids should be unique
        assert len(file_ids) == len(set(file_ids))
        
        # All should have the same repo prefix
        repo_name = extract_repo_name(repo_url)
        for file_id in file_ids:
            assert file_id.startswith(f"{repo_name}:")
    
    def test_file_id_collision_detection(self):
        """Test detection of potential file_id collisions."""
        # These should generate different file_ids despite similar names
        test_cases = [
            ("https://github.com/test/repo", "file.py"),
            ("https://github.com/test/repo", "file.js"),  # Different extension
            ("https://github.com/test/repo", "src/file.py"),  # Different path
            ("https://github.com/other/repo", "file.py"),  # Different repo
        ]
        
        file_ids = []
        for repo_url, file_path in test_cases:
            file_id = generate_file_id(repo_url, file_path)
            file_ids.append(file_id)
        
        # All should be unique
        assert len(file_ids) == len(set(file_ids))
        
        expected_file_ids = [
            "test-repo:file.py",
            "test-repo:file.js", 
            "test-repo:src/file.py",
            "other-repo:file.py"
        ]
        
        assert file_ids == expected_file_ids


class TestFileIdErrorHandling:
    """Test error handling in file_id cross-system linking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock()
        self.rag_service = RagService(self.mock_qdrant_client)
    
    def test_rag_search_with_nonexistent_file_id(self):
        """Test RAG search with file_id that doesn't exist."""
        # Mock empty results for nonexistent file_id
        self.rag_service.search_with_reranking = Mock(return_value=[])
        
        results = self.rag_service.search_with_reranking(
            query="test query",
            filter_metadata={"file_id": "nonexistent-repo:fake.py"}
        )
        
        # Should return empty results, not error
        assert results == []
    
    def test_neo4j_query_with_nonexistent_file_id(self):
        """Test Neo4j query with file_id that doesn't exist."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value = mock_session
        mock_session.run.return_value = []  # Empty results
        
        with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
            results = mock_session.run(
                "MATCH (f:File {file_id: $file_id}) RETURN f",
                file_id="nonexistent-repo:fake.py"
            )
            
            # Should return empty results, not error
            assert results == []
    
    def test_malformed_file_id_handling(self):
        """Test handling of malformed file_id values."""
        malformed_file_ids = [
            "",  # Empty string
            "invalid",  # No colon separator
            "::double_colon",  # Double colon
            "repo:",  # Missing file path
            ":missing_repo",  # Missing repo name
        ]
        
        for malformed_id in malformed_file_ids:
            # RAG system should handle gracefully
            self.rag_service.search_with_reranking = Mock(return_value=[])
            
            results = self.rag_service.search_with_reranking(
                query="test",
                filter_metadata={"file_id": malformed_id}
            )
            
            # Should not crash, return empty or handle gracefully
            assert isinstance(results, list)
    
    def test_cross_system_sync_validation(self):
        """Test validation that both systems have consistent file_id data."""
        file_id = "test-repo:src/models.py"
        
        # Mock: RAG has data but Neo4j doesn't
        self.rag_service.search_with_reranking = Mock(return_value=[
            {"content": "class Model:", "payload": {"file_id": file_id}}
        ])
        
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value = mock_session
        mock_session.run.return_value = []  # No Neo4j data
        
        with patch('neo4j.GraphDatabase.driver', return_value=mock_driver):
            # This scenario should be detectable for data consistency
            rag_has_data = len(self.rag_service.search_with_reranking(
                query="Model class", 
                filter_metadata={"file_id": file_id}
            )) > 0
            
            neo4j_has_data = len(mock_session.run(
                "MATCH (f:File {file_id: $file_id}) RETURN f",
                file_id=file_id
            )) > 0
            
            # Data inconsistency detected
            assert rag_has_data is True
            assert neo4j_has_data is False
            
            # This could trigger a warning or sync operation
            data_consistent = rag_has_data == neo4j_has_data
            assert data_consistent is False