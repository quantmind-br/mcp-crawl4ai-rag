"""
Python Tree-sitter Query Patterns

S-expression query patterns for extracting code structures from Python source code using Tree-sitter.
These queries are designed to extract the same information that the existing AST parser provides,
ensuring complete compatibility with the existing Neo4j schema and data structures.
"""

# Query to extract class definitions with names and body
PYTHON_QUERIES = {
    "classes": """
        (class_definition 
            name: (identifier) @class_name
            body: (block) @class_body) @class_def
    """,
    "functions": """
        (function_definition 
            name: (identifier) @func_name
            parameters: (parameters) @params
            body: (block) @func_body) @func_def
    """,
    "methods": """
        (class_definition
            body: (block
                (function_definition
                    name: (identifier) @method_name
                    parameters: (parameters) @method_params
                    body: (block) @method_body) @method_def)) @class_with_methods
    """,
    "imports": """
        (import_statement
            name: (dotted_name) @import_name) @import
    """,
    "import_from": """
        (import_from_statement
            module_name: (dotted_name) @module_name
            name: (dotted_name) @import_name) @import_from
    """,
    "import_from_multiple": """
        (import_from_statement
            module_name: (dotted_name) @module_name
            name: (import_list
                (dotted_name) @import_name)) @import_from_list
    """,
    "class_attributes": """
        (class_definition
            body: (block
                (expression_statement
                    (assignment
                        left: (identifier) @attr_name
                        right: (_) @attr_value)) @attr_assignment)) @class_with_attrs
    """,
    "class_inheritance": """
        (class_definition
            name: (identifier) @class_name
            superclasses: (argument_list
                (identifier) @base_class)) @inherited_class
    """,
    "async_functions": """
        (function_definition
            name: (identifier) @async_func_name
            parameters: (parameters) @async_params
            body: (block) @async_body) @async_func_def
    """,
    "decorators": """
        (decorated_definition
            (decorator
                (identifier) @decorator_name) @decorator
            definition: (_) @decorated_def) @decorated
    """,
    "docstrings": """
        (function_definition
            body: (block
                (expression_statement
                    (string) @docstring))) @func_with_docstring
    """,
    "class_docstrings": """
        (class_definition
            body: (block
                (expression_statement
                    (string) @class_docstring))) @class_with_docstring
    """,
}

# Pattern matching helpers for Python-specific constructs
PYTHON_HELPERS = {
    "parameter_patterns": {
        "simple": "(identifier) @param_name",
        "typed": "(typed_parameter (identifier) @param_name (type (_) @param_type))",
        "default": "(default_parameter (identifier) @param_name (_) @default_value)",
        "kwargs": "(dictionary_splat_pattern (identifier) @kwargs_name)",
        "varargs": "(list_splat_pattern (identifier) @varargs_name)",
    },
    "type_patterns": {
        "simple_type": "(identifier) @type_name",
        "generic_type": "(generic_type (identifier) @base_type)",
        "union_type": "(binary_operator left: (_) @left_type right: (_) @right_type)",
        "optional": "(subscript (identifier) @optional_base)",
    },
    "scope_patterns": {
        "module_level": "(module (_)*)",
        "class_level": "(class_definition body: (block (_)*))",
        "function_level": "(function_definition body: (block (_)*))",
    },
}

# Common Python language constructs that need special handling
PYTHON_CONSTRUCTS = {
    "magic_methods": [
        "__init__",
        "__new__",
        "__del__",
        "__repr__",
        "__str__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__len__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__call__",
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
    ],
    "property_decorators": [
        "property",
        "staticmethod",
        "classmethod",
        "getter",
        "setter",
        "deleter",
    ],
    "builtin_types": [
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "bytes",
        "bytearray",
        "frozenset",
        "complex",
        "object",
        "type",
    ],
    "common_imports": [
        "os",
        "sys",
        "json",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "typing",
        "pathlib",
        "logging",
        "unittest",
        "pytest",
    ],
}

# Query validation patterns to ensure extracted data is valid
VALIDATION_PATTERNS = {
    "valid_identifier": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    "valid_module": r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$",
    "private_method": r"^_[a-zA-Z0-9_]*$",
    "magic_method": r"^__[a-zA-Z0-9_]+__$",
    "constant": r"^[A-Z][A-Z0-9_]*$",
}


def get_query(query_type: str) -> str:
    """
    Get a specific query pattern by type.

    Args:
        query_type: The type of query to retrieve

    Returns:
        The S-expression query string

    Raises:
        KeyError: If query_type is not found
    """
    if query_type not in PYTHON_QUERIES:
        available = ", ".join(PYTHON_QUERIES.keys())
        raise KeyError(f"Query type '{query_type}' not found. Available: {available}")

    return PYTHON_QUERIES[query_type]


def get_all_queries() -> dict:
    """Get all available query patterns."""
    return PYTHON_QUERIES.copy()


def is_magic_method(method_name: str) -> bool:
    """Check if a method name is a Python magic method."""
    return method_name in PYTHON_CONSTRUCTS["magic_methods"]


def is_private_method(method_name: str) -> bool:
    """Check if a method name follows Python private naming convention."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["private_method"], method_name))


def validate_identifier(identifier: str) -> bool:
    """Validate that an identifier follows Python naming rules."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_identifier"], identifier))
