"""
Python file processor.

This module provides processing capabilities for Python files,
extracting docstrings from modules, functions, and classes using AST parsing.
"""

import ast
from typing import List

from .base_processor import BaseFileProcessor
from ..core.models import ProcessedContent
from ..core.exceptions import ProcessingError, SyntaxError as ProcessorSyntaxError


class PythonProcessor(BaseFileProcessor):
    """Process Python files using AST for docstring extraction."""

    def __init__(self):
        """Initialize Python processor."""
        super().__init__(
            name="python",
            supported_extensions=[".py", ".pyi"],
            max_file_size=1_000_000,  # 1MB limit for Python files
        )

    def _process_file_impl(
        self, file_path: str, relative_path: str, **kwargs
    ) -> List[ProcessedContent]:
        """
        Extract docstrings from Python file using AST.

        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
            **kwargs: Additional processing parameters

        Returns:
            List of ProcessedContent instances
        """
        try:
            source = self.read_file_content(file_path)

            # Parse the AST
            try:
                tree = ast.parse(source, filename=file_path)
            except SyntaxError as e:
                raise ProcessorSyntaxError(
                    f"Syntax error in Python file {file_path}: {e}",
                    file_path=file_path,
                    line_number=getattr(e, "lineno", None),
                )

            extracted_items = []

            # Module docstring
            module_doc = ast.get_docstring(tree, clean=True)
            if module_doc:
                processed_content = self.create_processed_content(
                    content=module_doc,
                    content_type="module",
                    name=relative_path,
                    signature=None,
                    line_number=1,
                    language="python",
                )
                extracted_items.append(processed_content)

            # Walk AST for functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node, clean=True)
                    if docstring:
                        processed_content = self.create_processed_content(
                            content=docstring,
                            content_type="function",
                            name=node.name,
                            signature=self._extract_signature(node),
                            line_number=node.lineno,
                            language="python",
                        )
                        extracted_items.append(processed_content)

                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node, clean=True)
                    if docstring:
                        processed_content = self.create_processed_content(
                            content=docstring,
                            content_type="class",
                            name=node.name,
                            signature=None,
                            line_number=node.lineno,
                            language="python",
                        )
                        extracted_items.append(processed_content)

            return extracted_items

        except ProcessorSyntaxError:
            # Re-raise syntax errors
            raise
        except Exception as e:
            raise ProcessingError(
                f"Error processing Python file {file_path}: {e}",
                file_path=file_path,
                processor_name=self.name,
            )

    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """
        Extract function signature with type annotations.

        Args:
            node: AST function definition node

        Returns:
            Function signature string
        """
        try:
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    try:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    except Exception:
                        # Fallback if unparse fails
                        arg_str += ": <annotation>"
                args.append(arg_str)

            signature = f"({', '.join(args)})"
            if node.returns:
                try:
                    signature += f" -> {ast.unparse(node.returns)}"
                except Exception:
                    # Fallback if unparse fails
                    signature += " -> <return_type>"

            return signature
        except Exception:
            return "(signature_extraction_failed)"
