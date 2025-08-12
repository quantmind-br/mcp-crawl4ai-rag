from src.utils.file_id_generator import generate_file_id, validate_file_id


class TestFileIdGenerator:
    def test_generate_file_id_basic(self):
        """Test that generate_file_id creates a valid file ID"""
        repo_url = "https://github.com/test/repo"
        file_path = "test/path/file.py"
        file_id = generate_file_id(repo_url, file_path)
        assert isinstance(file_id, str)
        assert len(file_id) > 0
        assert validate_file_id(file_id)

    def test_generate_file_id_uniqueness(self):
        """Test that generate_file_id creates unique IDs for different paths"""
        repo_url = "https://github.com/test/repo"
        file_path1 = "test/path/file1.py"
        file_path2 = "test/path/file2.py"
        id1 = generate_file_id(repo_url, file_path1)
        id2 = generate_file_id(repo_url, file_path2)
        assert id1 != id2

    def test_generate_file_id_consistency(self):
        """Test that generate_file_id creates consistent IDs for same path"""
        repo_url = "https://github.com/test/repo"
        file_path = "test/path/file.py"
        id1 = generate_file_id(repo_url, file_path)
        id2 = generate_file_id(repo_url, file_path)
        assert id1 == id2

    def test_validate_file_id_valid(self):
        """Test that validate_file_id returns True for valid IDs"""
        repo_url = "https://github.com/test/repo"
        file_path = "test/path/file.py"
        file_id = generate_file_id(repo_url, file_path)
        assert validate_file_id(file_id)

    def test_validate_file_id_invalid(self):
        """Test that validate_file_id returns False for invalid IDs"""
        assert not validate_file_id(None)
        assert not validate_file_id("")
        assert not validate_file_id("invalid-id")
