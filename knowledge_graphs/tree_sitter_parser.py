"""
Tree-sitter Parser Implementation

Main implementation of the Tree-sitter-based multi-language parser. This class provides
the core functionality for parsing source code using Tree-sitter and extracting structural
information that matches the existing AST parser output format exactly.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

# Tree-sitter imports
import tree_sitter
from tree_sitter import Language, Parser, Node

# Local imports
from knowledge_graphs.language_parser import (
    LanguageParser,
    ParseResult,
    ParsedClass,
    ParsedFunction,
    ParsedMethod,
    ParsedAttribute,
)

# Configure logging
logger = logging.getLogger(__name__)


class TreeSitterParser(LanguageParser):
    """
    Tree-sitter-based multi-language parser implementation.

    This parser uses Tree-sitter grammars and S-expression queries to extract
    structural information from source code in multiple programming languages.
    The output is designed to exactly match the format expected by the existing
    Neo4j population logic and downstream consumers.
    """

    def __init__(self, language_name: str):
        """
        Initialize the Tree-sitter parser for a specific language.

        Args:
            language_name: Programming language name (e.g., 'python', 'typescript')
        """
        super().__init__(language_name)

        # Language-specific configuration
        self._setup_language_config()

        # Tree-sitter components
        self.parser: Optional[Parser] = None
        self.ts_language: Optional[Language] = None

        # Query cache for performance
        self._query_cache: Dict[str, Query] = {}

        # Initialize Tree-sitter components
        self._initialize_treesitter()

        # Statistics
        self.stats = {
            "files_parsed": 0,
            "parse_errors": 0,
            "query_cache_hits": 0,
            "classes_extracted": 0,
            "functions_extracted": 0,
            "methods_extracted": 0,
        }

        logger.debug(f"TreeSitterParser initialized for language: {language_name}")

    @property
    def language(self) -> str:
        """Get the language name as string (for compatibility with tests)."""
        return self.language_name

    def _setup_language_config(self):
        """Setup language-specific configuration and file extensions."""
        extension_map = {
            "python": [".py", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs", ".cjs"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "c": [".c", ".h"],
            "cpp": [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
            "c_sharp": [".cs"],
            "php": [".php", ".php3", ".php4", ".php5", ".phtml"],
            "ruby": [".rb", ".rbw"],
            "kotlin": [".kt", ".kts"],
        }

        self.supported_extensions = extension_map.get(self.language_name, [])

    def _initialize_treesitter(self):
        """Initialize Tree-sitter parser and language for this language."""
        try:
            # Import the language module dynamically
            language_module_map = {
                "python": "tree_sitter_python",
                "javascript": "tree_sitter_javascript",
                "typescript": "tree_sitter_typescript",
                "java": "tree_sitter_java",
                "go": "tree_sitter_go",
                "rust": "tree_sitter_rust",
                "c": "tree_sitter_c",
                "cpp": "tree_sitter_cpp",
                "c_sharp": "tree_sitter_c_sharp",
                "php": "tree_sitter_php",
                "ruby": "tree_sitter_ruby",
                "kotlin": "tree_sitter_kotlin",
            }

            language_function_map = {
                "python": "language",
                "javascript": "language",
                "typescript": "language_typescript",  # Special case
                "java": "language",
                "go": "language",
                "rust": "language",
                "c": "language",
                "cpp": "language",
                "c_sharp": "language",
                "php": "language_php",  # Special case
                "ruby": "language",
                "kotlin": "language",
            }

            if self.language_name not in language_module_map:
                raise ValueError(f"Unsupported language: {self.language_name}")

            # Import the language module
            module_name = language_module_map[self.language_name]
            function_name = language_function_map[self.language_name]

            module = __import__(module_name)
            language_function = getattr(module, function_name)
            language_capsule = language_function()

            # Create Language and Parser objects
            self.ts_language = Language(language_capsule)
            self.parser = Parser()
            self.parser.language = self.ts_language

            logger.debug(
                f"Tree-sitter initialized successfully for {self.language_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize Tree-sitter for {self.language_name}: {e}"
            )
            raise

    def _get_query(self, query_type: str) -> Optional[tree_sitter.Query]:
        """
        Get a compiled Tree-sitter query with caching.

        Args:
            query_type: Type of query (e.g., 'classes', 'functions')

        Returns:
            Compiled Query object or None if not available
        """
        cache_key = f"{self.language_name}:{query_type}"

        if cache_key in self._query_cache:
            self.stats["query_cache_hits"] += 1
            return self._query_cache[cache_key]

        # S-expression queries are deprecated - using manual traversal instead
        logger.debug(
            f"Query system deprecated for {cache_key} - using manual traversal"
        )
        return None

    def parse(self, file_content: str, file_path: str) -> ParseResult:
        """
        Parse source code and extract structural information.

        Args:
            file_content: The source code content as a string
            file_path: Path to the source file

        Returns:
            ParseResult containing extracted classes, functions, imports, etc.
        """
        if not self.parser or not self.ts_language:
            return self._create_empty_result(file_path, ["Parser not initialized"])

        try:
            # Parse the source code
            tree = self.parser.parse(file_content.encode("utf-8"))

            if not tree.root_node:
                return self._create_empty_result(
                    file_path, ["Failed to parse - no root node"]
                )

            # Extract structural information
            parsed_classes = self._extract_classes(tree.root_node, file_content)
            parsed_functions = self._extract_functions(tree.root_node, file_content)
            imports = self._extract_imports(tree.root_node, file_content)

            # Convert to dictionary format expected by Neo4j
            classes_dict, functions_dict = self._convert_to_dict_structure(
                parsed_classes, parsed_functions
            )

            # Update statistics
            self.stats["files_parsed"] += 1
            self.stats["classes_extracted"] += len(classes_dict)
            self.stats["functions_extracted"] += len(functions_dict)
            self.stats["methods_extracted"] += sum(
                len(cls.methods) for cls in parsed_classes
            )

            # Create result
            result = ParseResult(
                module_name=self._extract_module_name(file_path),
                file_path=file_path,
                classes=classes_dict,
                functions=functions_dict,
                imports=imports,
                line_count=len(file_content.splitlines()),
                language=self.language_name,
                errors=[],
            )

            logger.debug(
                f"Successfully parsed {file_path}: {len(classes_dict)} classes, {len(functions_dict)} functions"
            )
            return result

        except Exception as e:
            self.stats["parse_errors"] += 1
            logger.error(f"Error parsing {file_path}: {e}")
            return self._create_empty_result(file_path, [f"Parse error: {str(e)}"])

    def _extract_classes(
        self, root_node: Node, source_content: str
    ) -> List[ParsedClass]:
        """Extract class information from the parse tree."""
        classes: List[ParsedClass] = []

        # No need for queries anymore - using manual traversal

        # Manual tree traversal approach (Tree-sitter Python bindings don't support S-expression queries properly)
        def find_nodes_by_type(node, node_types):
            """Find all nodes of specified types in the tree."""
            results = []
            if isinstance(node_types, str):
                node_types = [node_types]
            if node.type in node_types:
                results.append(node)
            for child in node.children:
                results.extend(find_nodes_by_type(child, node_types))
            return results

        # Find class-like structures based on language
        class_types = self._get_class_node_types()
        class_nodes = find_nodes_by_type(root_node, class_types)

        for class_node in class_nodes:
            class_info = self._extract_single_class(class_node, source_content)
            if class_info:
                classes.append(class_info)

        return classes

    def _get_class_node_types(self) -> List[str]:
        """Get the node types that represent classes in this language."""
        class_type_map = {
            "python": ["class_definition"],
            "javascript": ["class_declaration"],
            "typescript": ["class_declaration"],
            "java": ["class_declaration"],
            "go": ["type_declaration"],  # Go structs/interfaces
            "rust": ["struct_item", "enum_item", "trait_item", "impl_item"],
            "c": ["struct_specifier"],
            "cpp": ["class_specifier", "struct_specifier"],
            "c_sharp": [
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
            ],
            "php": ["class_declaration"],
            "ruby": ["class"],
            "kotlin": ["class_declaration"],
        }
        return class_type_map.get(self.language_name, ["class_declaration"])

    def _get_function_node_types(self) -> List[str]:
        """Get the node types that represent functions in this language."""
        function_type_map = {
            "python": ["function_definition"],
            "javascript": ["function_declaration"],
            "typescript": ["function_declaration"],
            "java": [
                "method_declaration"
            ],  # Java has methods, not standalone functions
            "go": ["function_declaration", "method_declaration"],
            "rust": ["function_item"],
            "c": ["function_definition"],
            "cpp": ["function_definition"],
            "c_sharp": ["method_declaration"],
            "php": ["function_declaration"],
            "ruby": ["method"],
            "kotlin": ["function_declaration"],
        }
        return function_type_map.get(self.language_name, ["function_declaration"])

    def _find_class_definition_node(self, name_node: Node) -> Optional[Node]:
        """Find the class definition node from a class name node."""
        current = name_node.parent
        while current:
            if current.type in ["class_definition", "class_declaration"]:
                return current
            current = current.parent
        return name_node.parent  # fallback to immediate parent

    def _find_function_definition_node(self, name_node: Node) -> Optional[Node]:
        """Find the function definition node from a function name node."""
        current = name_node.parent
        while current:
            if current.type in [
                "function_definition",
                "function_declaration",
                "method_definition",
            ]:
                return current
            current = current.parent
        return name_node.parent  # fallback to immediate parent

    def _extract_name_from_definition_node(
        self, def_node: Node, definition_type: str
    ) -> Optional[str]:
        """Extract the name identifier from a definition node."""
        # Look for identifier nodes in the definition
        for child in def_node.children:
            if child.type == "identifier":
                return self._get_node_text(
                    child, ""
                )  # Use empty string as we're using node.text

        # Fallback: recursively search for first identifier
        def find_identifier(node: Node) -> Optional[str]:
            if node.type == "identifier":
                return self._get_node_text(node, "")
            for child in node.children:
                result = find_identifier(child)
                if result:
                    return result
            return None

        return find_identifier(def_node)

    def _is_method_inside_class(self, func_node: Node) -> bool:
        """Check if a function node is inside a class definition."""
        current = func_node.parent
        while current:
            if current.type in ["class_definition", "class_declaration"]:
                return True
            current = current.parent
        return False

    def _extract_single_class(
        self, class_node: Node, source_content: str
    ) -> Optional[ParsedClass]:
        """Extract information from a single class node."""
        try:
            # Get class name from the node
            class_name = self._extract_name_from_definition_node(class_node, "class")
            if not class_name:
                return None

            # Extract methods
            methods = self._extract_methods_from_class(class_node, source_content)

            # Extract attributes/fields
            attributes = self._extract_attributes_from_class(class_node, source_content)

            # Get position information
            start_line = class_node.start_point[0] + 1
            end_line = class_node.end_point[0] + 1

            # Create full name (could include namespace/package info later)
            full_name = class_name

            return ParsedClass(
                name=class_name,
                full_name=full_name,
                methods=methods,
                attributes=attributes,
                line_start=start_line,
                line_end=end_line,
                docstring=self._extract_docstring(class_node, source_content),
            )

        except Exception as e:
            logger.debug(f"Error extracting class: {e}")
            return None

    def _extract_methods_from_class(
        self, class_node: Node, source_content: str
    ) -> List[ParsedMethod]:
        """Extract methods from a class node."""
        methods: List[ParsedMethod] = []

        # Manual traversal to find method nodes within the class
        def find_method_nodes(node):
            method_nodes = []
            for child in node.children:
                if child.type in [
                    "method_declaration",
                    "method_definition",
                    "function_definition",
                ]:
                    method_nodes.append(child)
                # Recursively search in class bodies
                elif child.type in ["class_body", "block"]:
                    method_nodes.extend(find_method_nodes(child))
            return method_nodes

        method_nodes = find_method_nodes(class_node)
        for method_node in method_nodes:
            method_info = self._extract_single_method(method_node, source_content)
            if method_info:
                methods.append(method_info)

        return methods

    def _extract_single_method(
        self, method_node: Node, source_content: str
    ) -> Optional[ParsedMethod]:
        """Extract information from a single method node."""
        try:
            # Get method name
            method_name = self._extract_name_from_definition_node(method_node, "method")
            if not method_name:
                return None

            # Extract parameters
            params = self._extract_parameters(method_node, source_content)

            # Extract return type (if available)
            return_type = (
                self._extract_return_type(method_node, source_content) or "void"
            )

            # Get position information
            start_line = method_node.start_point[0] + 1
            end_line = method_node.end_point[0] + 1

            return ParsedMethod(
                name=method_name,
                params=params,
                return_type=return_type,
                line_start=start_line,
                line_end=end_line,
                docstring=self._extract_docstring(method_node, source_content),
            )

        except Exception as e:
            logger.debug(f"Error extracting method: {e}")
            return None

    def _extract_functions(
        self, root_node: Node, source_content: str
    ) -> List[ParsedFunction]:
        """Extract function information from the parse tree."""
        functions: List[ParsedFunction] = []

        # Manual tree traversal approach
        def find_nodes_by_type(node, node_types):
            """Find all nodes of specified types in the tree."""
            results = []
            if isinstance(node_types, str):
                node_types = [node_types]
            if node.type in node_types:
                results.append(node)
            for child in node.children:
                results.extend(find_nodes_by_type(child, node_types))
            return results

        # Find function-like structures based on language
        function_types = self._get_function_node_types()
        function_nodes = find_nodes_by_type(root_node, function_types)

        for func_node in function_nodes:
            # Skip functions that are inside classes (those are methods)
            if not self._is_method_inside_class(func_node):
                func_info = self._extract_single_function(func_node, source_content)
                if func_info:
                    functions.append(func_info)

        return functions

    def _extract_single_function(
        self, func_node: Node, source_content: str
    ) -> Optional[ParsedFunction]:
        """Extract information from a single function node."""
        try:
            # Get function name
            func_name = self._extract_name_from_definition_node(func_node, "function")
            if not func_name:
                return None

            # Extract parameters
            params = self._extract_parameters(func_node, source_content)

            # Extract return type
            return_type = self._extract_return_type(func_node, source_content) or "void"

            # Get position information
            start_line = func_node.start_point[0] + 1
            end_line = func_node.end_point[0] + 1

            # Create full name
            full_name = func_name

            return ParsedFunction(
                name=func_name,
                full_name=full_name,
                params=params,
                return_type=return_type,
                line_start=start_line,
                line_end=end_line,
                docstring=self._extract_docstring(func_node, source_content),
            )

        except Exception as e:
            logger.debug(f"Error extracting function: {e}")
            return None

    def _extract_attributes_from_class(
        self, class_node: Node, source_content: str
    ) -> List[ParsedAttribute]:
        """Extract attributes/fields from a class node."""
        attributes = []

        # Try different query types for different languages
        # Manual traversal to find attribute/field nodes within the class
        def find_attribute_nodes(node):
            attr_nodes = []
            for child in node.children:
                if child.type in [
                    "field_declaration",
                    "variable_declarator",
                    "assignment_expression",
                    "expression_statement",
                ]:
                    attr_nodes.append(child)
                # Recursively search in class bodies
                elif child.type in ["class_body", "block"]:
                    attr_nodes.extend(find_attribute_nodes(child))
            return attr_nodes

        attr_nodes = find_attribute_nodes(class_node)
        for attr_node in attr_nodes:
            attr_info = self._extract_single_attribute(attr_node, source_content)
            if attr_info:
                attributes.append(attr_info)

        return attributes

    def _extract_single_attribute(
        self, attr_node: Node, source_content: str
    ) -> Optional[ParsedAttribute]:
        """Extract information from a single attribute node."""
        try:
            # Get attribute name - try different capture patterns
            attr_name = None
            for pattern in ["attr_name", "field_name", "prop_name"]:
                attr_name = self._get_node_text(
                    attr_node.parent, source_content, pattern
                )
                if attr_name:
                    break

            if not attr_name:
                return None

            # Extract type information
            attr_type = (
                self._extract_attribute_type(attr_node.parent, source_content)
                or "unknown"
            )

            # Get position information
            start_line = attr_node.start_point[0] + 1
            end_line = attr_node.end_point[0] + 1

            return ParsedAttribute(
                name=attr_name, type=attr_type, line_start=start_line, line_end=end_line
            )

        except Exception as e:
            logger.debug(f"Error extracting attribute: {e}")
            return None

    def _extract_imports(self, root_node: Node, source_content: str) -> List[str]:
        """Extract import statements from the parse tree."""
        imports = set()

        # Manual tree traversal to find import nodes
        def find_nodes_by_type(node, node_types):
            """Find all nodes of specified types in the tree."""
            results = []
            if isinstance(node_types, str):
                node_types = [node_types]
            if node.type in node_types:
                results.append(node)
            for child in node.children:
                results.extend(find_nodes_by_type(child, node_types))
            return results

        # Language-specific import node types
        import_types = self._get_import_node_types()
        import_nodes = find_nodes_by_type(root_node, import_types)

        for import_node in import_nodes:
            import_name = self._extract_import_name(import_node, source_content)
            if import_name:
                imports.add(import_name)

        return list(imports)

    def _get_import_node_types(self) -> List[str]:
        """Get the node types that represent imports in this language."""
        import_type_map = {
            "python": ["import_statement", "import_from_statement"],
            "javascript": ["import_statement"],
            "typescript": ["import_statement"],
            "java": ["import_declaration"],
            "go": ["import_declaration", "import_spec"],
            "rust": ["use_declaration"],
            "c": ["preproc_include"],
            "cpp": ["preproc_include"],
            "c_sharp": ["using_directive"],
            "php": ["use_declaration", "include_expression", "require_expression"],
            "ruby": ["load", "require"],
            "kotlin": ["import_header"],
        }
        return import_type_map.get(self.language_name, ["import_statement"])

    def _extract_import_name(
        self, import_node: Node, source_content: str
    ) -> Optional[str]:
        """Extract the import name from an import node."""

        # Find string literals or identifiers in the import
        def find_import_target(node):
            if node.type in [
                "string_literal",
                "interpreted_string_literal",
                "raw_string_literal",
            ]:
                text = self._get_node_text(node, source_content)
                return self._clean_import_name(text)
            elif node.type == "identifier":
                return self._get_node_text(node, source_content)
            elif node.type in ["scoped_identifier", "dotted_name"]:
                return self._get_node_text(node, source_content)

            for child in node.children:
                result = find_import_target(child)
                if result:
                    return result
            return None

        return find_import_target(import_node)

    def _extract_parameters(self, func_node: Node, source_content: str) -> List[str]:
        """Extract parameter list from a function or method node."""
        params = []

        # Find parameter nodes
        for child in func_node.children:
            if child.type in ["parameters", "formal_parameters", "parameter_list"]:
                for param_child in child.children:
                    if param_child.type in [
                        "parameter",
                        "parameter_declaration",
                        "identifier",
                    ]:
                        param_text = self._get_node_text(param_child, source_content)
                        if param_text and param_text not in ["(", ")", ","]:
                            params.append(param_text.strip())

        return params

    def _extract_return_type(
        self, func_node: Node, source_content: str
    ) -> Optional[str]:
        """Extract return type from a function node."""
        # Language-specific return type extraction
        for child in func_node.children:
            if child.type in ["type_annotation", "return_type"]:
                return self._get_node_text(child, source_content)

        return None

    def _extract_attribute_type(
        self, attr_node: Node, source_content: str
    ) -> Optional[str]:
        """Extract type information from an attribute node."""
        for child in attr_node.children:
            if child.type in ["type", "type_annotation"]:
                return self._get_node_text(child, source_content)

        return None

    def _extract_docstring(self, node: Node, source_content: str) -> Optional[str]:
        """Extract docstring/comment from a node."""
        # This is language-specific and would need more sophisticated implementation
        return None

    def _get_node_text(
        self, node: Node, source_content: str, capture_name: Optional[str] = None
    ) -> Optional[str]:
        """Extract text content from a Tree-sitter node."""
        if not node:
            return None

        try:
            # Use the node.text property which returns bytes
            node_text = node.text.decode("utf-8")
            return node_text

        except Exception as e:
            logger.debug(f"Error extracting node text: {e}")
            # Fallback to manual byte extraction
            try:
                start_byte = node.start_byte
                end_byte = node.end_byte

                if start_byte >= len(source_content.encode("utf-8")) or end_byte > len(
                    source_content.encode("utf-8")
                ):
                    return None

                # Extract text using byte positions
                node_bytes = source_content.encode("utf-8")[start_byte:end_byte]
                return node_bytes.decode("utf-8")

            except Exception as e2:
                logger.debug(f"Fallback node text extraction failed: {e2}")
                return None

    def _clean_import_name(self, import_name: str) -> Optional[str]:
        """Clean and normalize import names."""
        if not import_name:
            return None

        # Remove quotes and common punctuation
        cleaned = import_name.strip().strip("\"'();")

        # Language-specific cleaning
        if self.language_name == "python":
            # Remove 'import' keyword if present
            if cleaned.startswith("import "):
                cleaned = cleaned[7:]
            elif cleaned.startswith("from "):
                # For 'from x import y', extract y
                parts = cleaned.split(" import ")
                if len(parts) > 1:
                    cleaned = parts[1]

        return cleaned if cleaned else None

    def _create_empty_result(self, file_path: str, errors: List[str]) -> ParseResult:
        """Create an empty ParseResult for failed parsing."""
        return ParseResult(
            module_name=self._extract_module_name(file_path),
            file_path=file_path,
            classes=[],
            functions=[],
            imports=[],
            line_count=0,
            language=self.language_name,
            errors=errors,
        )

    def is_supported_file(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def get_stats(self) -> Dict[str, int]:
        """Get parser statistics."""
        return self.stats.copy()

    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.debug(f"Cleared query cache for {self.language_name}")
