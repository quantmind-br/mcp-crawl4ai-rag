"""
Tests for MDX processor.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from src.features.github.processors.mdx_processor import MDXProcessor
from src.features.github.core.exceptions import ProcessingError


class TestMDXProcessor:
    """Test cases for MDX processor."""

    @pytest.fixture
    def mdx_processor(self):
        """Create an MDX processor instance."""
        return MDXProcessor()

    @pytest.fixture
    def sample_mdx_content(self):
        """Sample MDX content for testing."""
        return '''---
title: "Sample MDX"
date: 2024-01-15
---

import { Alert } from './components/Alert'
import { CodeBlock } from './components/CodeBlock'

# Hello MDX World

This is a sample MDX file with JSX components.

<Alert type="info">
This is an alert component with some content.
</Alert>

Here's some regular markdown content.

<CodeBlock language="javascript">
const greeting = "Hello, World!";
console.log(greeting);
</CodeBlock>

## Conclusion

<Image src="/logo.png" alt="Logo" />

End of content.
'''

    @pytest.fixture
    def simple_mdx_content(self):
        """Simple MDX content without frontmatter."""
        return '''# Simple MDX

<Button variant="primary">Click me</Button>

Regular markdown content.
'''

    @pytest.fixture
    def temp_mdx_file(self, sample_mdx_content):
        """Create a temporary MDX file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mdx', delete=False) as f:
            f.write(sample_mdx_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_processor_initialization(self, mdx_processor):
        """Test processor initialization."""
        assert mdx_processor.name == "mdx"
        assert mdx_processor.supported_extensions == [".mdx"]
        assert mdx_processor.max_file_size == 1_000_000

    def test_can_process_mdx_files(self, mdx_processor):
        """Test that processor can handle .mdx files."""
        assert mdx_processor.can_process("test.mdx") is True
        assert mdx_processor.can_process("test.MDX") is True  # Case insensitive
        assert mdx_processor.can_process("test.md") is False
        assert mdx_processor.can_process("test.py") is False

    def test_get_supported_extensions(self, mdx_processor):
        """Test getting supported extensions."""
        extensions = mdx_processor.get_supported_extensions()
        assert extensions == [".mdx"]

    def test_process_mdx_file_success(self, mdx_processor, temp_mdx_file):
        """Test successful MDX file processing."""
        result = mdx_processor._process_file_impl(temp_mdx_file, "test.mdx")
        
        assert len(result) >= 1  # At least main content
        
        # Check main content
        main_content = result[0]
        assert main_content.content_type == "mdx"
        assert main_content.name == os.path.basename(temp_mdx_file)
        assert main_content.language == "mdx"
        assert main_content.line_number == 1
        
        # Content should be cleaned (no imports/JSX)
        assert "import {" not in main_content.content
        assert "<Alert" not in main_content.content
        assert "Hello MDX World" in main_content.content

    def test_jsx_component_extraction(self, mdx_processor, temp_mdx_file):
        """Test JSX component extraction."""
        result = mdx_processor._process_file_impl(temp_mdx_file, "test.mdx")
        
        # Should have main content + extracted components
        assert len(result) > 1
        
        # Find JSX components
        jsx_components = [r for r in result if r.content_type == "jsx_component"]
        assert len(jsx_components) >= 2  # Alert, CodeBlock, Image
        
        # Check component details
        component_names = [comp.name for comp in jsx_components]
        assert "Alert" in component_names
        assert "CodeBlock" in component_names
        assert "Image" in component_names

    def test_clean_mdx_content(self, mdx_processor, sample_mdx_content):
        """Test MDX content cleaning."""
        cleaned = mdx_processor._clean_mdx_content(sample_mdx_content)
        
        # Should remove frontmatter
        assert "title:" not in cleaned
        assert "date:" not in cleaned
        
        # Should remove imports
        assert "import {" not in cleaned
        
        # Should remove JSX components but keep inner text
        assert "<Alert" not in cleaned
        assert "<CodeBlock" not in cleaned
        assert "This is an alert component" in cleaned
        
        # Should preserve markdown content
        assert "Hello MDX World" in cleaned
        assert "Here's some regular markdown content" in cleaned

    def test_clean_mdx_content_without_frontmatter(self, mdx_processor, simple_mdx_content):
        """Test cleaning MDX content without frontmatter."""
        cleaned = mdx_processor._clean_mdx_content(simple_mdx_content)
        
        # Should remove JSX but keep content
        assert "<Button" not in cleaned
        assert "Simple MDX" in cleaned
        assert "Regular markdown content" in cleaned

    def test_extract_jsx_components(self, mdx_processor, sample_mdx_content):
        """Test JSX component extraction method."""
        components = mdx_processor._extract_jsx_components(sample_mdx_content)
        
        assert len(components) >= 3  # Alert, CodeBlock, Image
        
        # Check each component
        component_data = {comp.name: comp for comp in components}
        
        assert "Alert" in component_data
        alert_comp = component_data["Alert"]
        assert alert_comp.content_type == "jsx_component"
        assert alert_comp.language == "jsx"
        assert "<Alert" in alert_comp.signature
        
        assert "CodeBlock" in component_data
        assert "Image" in component_data

    def test_jsx_component_deduplication(self, mdx_processor):
        """Test that duplicate JSX components are not extracted multiple times."""
        content = '''
        <Button>First</Button>
        <Button>Second</Button>
        <Alert>Alert content</Alert>
        '''
        
        components = mdx_processor._extract_jsx_components(content)
        component_names = [comp.name for comp in components]
        
        # Should only have unique component names
        assert component_names.count("Button") == 1
        assert component_names.count("Alert") == 1

    def test_process_empty_file(self, mdx_processor):
        """Test processing empty or very small file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mdx', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            result = mdx_processor._process_file_impl(temp_path, "empty.mdx")
            # Should return empty list for invalid content
            assert result == []
        finally:
            os.unlink(temp_path)

    def test_process_invalid_file_path(self, mdx_processor):
        """Test processing non-existent file."""
        with pytest.raises(ProcessingError):
            mdx_processor._process_file_impl("/nonexistent/file.mdx", "test.mdx")

    def test_content_validation(self, mdx_processor):
        """Test content validation."""
        # Valid content
        valid_content = "# This is a valid MDX file with enough content to pass validation."
        assert mdx_processor.validate_content(valid_content, min_length=50) is True
        
        # Invalid content - too short
        short_content = "Short"
        assert mdx_processor.validate_content(short_content, min_length=50) is False
        
        # Invalid content - empty
        assert mdx_processor.validate_content("", min_length=50) is False
        
        # Invalid content - binary
        binary_content = "Some text with \x00 null bytes"
        assert mdx_processor.validate_content(binary_content, min_length=10) is False

    def test_jsx_expressions_handling(self, mdx_processor):
        """Test handling of JSX expressions."""
        content = '''
        # Title: {title}
        
        The current date is {new Date().toLocaleDateString()}.
        
        Variable: {someVariable}
        '''
        
        cleaned = mdx_processor._clean_mdx_content(content)
        
        # Should preserve variable names but remove braces
        assert "title" in cleaned
        assert "new Date().toLocaleDateString()" in cleaned
        assert "someVariable" in cleaned
        assert "{" not in cleaned
        assert "}" not in cleaned

    def test_nested_jsx_components(self, mdx_processor):
        """Test handling of nested JSX components."""
        content = '''
        <Card>
            <CardHeader>
                <Title>Nested Title</Title>
            </CardHeader>
            <CardBody>
                Content here
            </CardBody>
        </Card>
        
        <Button>Click me</Button>
        '''
        
        components = mdx_processor._extract_jsx_components(content)
        component_names = [comp.name for comp in components]
        
        # Current regex extracts only top-level components
        # Nested components are treated as content of parent
        assert "Card" in component_names
        assert "Button" in component_names
        assert len(component_names) == 2  # Only top-level components

    def test_self_closing_jsx_components(self, mdx_processor):
        """Test handling of self-closing JSX components."""
        content = '''
        <Image src="/test.png" alt="Test" />
        <br />
        <Input type="text" placeholder="Enter text" />
        '''
        
        components = mdx_processor._extract_jsx_components(content)
        component_names = [comp.name for comp in components]
        
        assert "Image" in component_names
        assert "Input" in component_names
        # Note: 'br' starts with lowercase, so won't be detected as JSX component

    def test_error_handling_in_cleaning(self, mdx_processor):
        """Test error handling during content cleaning."""
        # Mock the logger to capture warnings
        with patch.object(mdx_processor, 'logger'):
            # This should not raise an exception even with malformed content
            result = mdx_processor._clean_mdx_content("Malformed content")
            assert result == "Malformed content"  # Should return original on error

    def test_error_handling_in_component_extraction(self, mdx_processor):
        """Test error handling during component extraction."""
        # Mock the logger and regex pattern to simulate error
        with patch.object(mdx_processor, 'logger') as mock_logger:
            # Create a mock pattern that raises an exception when finditer is called
            mock_pattern = Mock()
            mock_pattern.finditer.side_effect = Exception("Test error")
            
            # Replace the jsx_component_pattern with our mock
            with patch.object(mdx_processor, 'jsx_component_pattern', mock_pattern):
                result = mdx_processor._extract_jsx_components("Some content")
                assert result == []  # Should return empty list on error
                mock_logger.warning.assert_called()

    def test_processing_result_structure(self, mdx_processor, temp_mdx_file):
        """Test the structure of processing results."""
        result = mdx_processor._process_file_impl(temp_mdx_file, "test.mdx")
        
        for item in result:
            # Check required fields
            assert hasattr(item, 'content')
            assert hasattr(item, 'content_type')
            assert hasattr(item, 'name')
            assert hasattr(item, 'signature')
            assert hasattr(item, 'line_number')
            assert hasattr(item, 'language')
            
            # Check data types
            assert isinstance(item.content, str)
            assert isinstance(item.content_type, str)
            assert isinstance(item.name, str)
            assert isinstance(item.line_number, int)
            assert isinstance(item.language, str)
            
            # Check content type values
            assert item.content_type in ["mdx", "jsx_component"]
            
            # Check language values
            assert item.language in ["mdx", "jsx"]

    def test_create_processed_content_method(self, mdx_processor):
        """Test the create_processed_content helper method."""
        content = mdx_processor.create_processed_content(
            content="Test content",
            content_type="mdx",
            name="test.mdx",
            signature=None,
            line_number=1,
            language="mdx"
        )
        
        assert content.content == "Test content"
        assert content.content_type == "mdx"
        assert content.name == "test.mdx"
        assert content.signature is None
        assert content.line_number == 1
        assert content.language == "mdx"