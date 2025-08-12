"""
Go Tree-sitter Query Patterns

S-expression query patterns for extracting code structures from Go source code using Tree-sitter.
These queries handle Go-specific constructs like packages, interfaces, struct methods,
and goroutines while maintaining compatibility with the existing Neo4j schema.
"""

# Query patterns for Go language constructs
GO_QUERIES = {
    # Map Go structs and interfaces to "classes" for Neo4j compatibility
    "classes": """
        (type_declaration
            (type_spec
                name: (type_identifier) @class_name
                type: (struct_type
                    (field_declaration_list) @class_fields))) @class_definition
                    
        (type_declaration
            (type_spec
                name: (type_identifier) @class_name
                type: (interface_type
                    (method_spec_list) @class_methods))) @class_definition
    """,
    "functions": """
        (function_declaration
            name: (identifier) @func_name
            parameters: (parameter_list) @func_parameters
            body: (block) @func_body) @func_definition
    """,
    "methods": """
        (method_declaration
            receiver: (parameter_list 
                (parameter_declaration
                    name: (identifier) @receiver_name
                    type: (_) @receiver_type)) @receiver
            name: (identifier) @method_name
            parameters: (parameter_list) @method_parameters
            body: (block) @method_body) @method_definition
    """,
    "structs": """
        (type_declaration
            (type_spec
                name: (type_identifier) @struct_name
                type: (struct_type
                    (field_declaration_list) @struct_fields))) @struct_def
    """,
    "interfaces": """
        (type_declaration
            (type_spec
                name: (type_identifier) @interface_name
                type: (interface_type
                    (method_spec_list) @interface_methods))) @interface_def
    """,
    "struct_fields": """
        (struct_type
            (field_declaration_list
                (field_declaration
                    name: (field_identifier) @field_name
                    type: (_) @field_type) @field_decl)) @struct_with_fields
    """,
    "imports": """
        (import_declaration
            (import_spec
                path: (interpreted_string_literal) @import_source)) @import_definition
    """,
    "named_imports": """
        (import_declaration
            (import_spec
                name: (package_identifier) @import_alias
                path: (interpreted_string_literal) @import_path)) @named_import
    """,
    "constants": """
        (const_declaration
            (const_spec
                name: (identifier) @const_name
                type: (_) @const_type
                value: (_) @const_value)) @const_def
    """,
    "variables": """
        (var_declaration
            (var_spec
                name: (identifier) @var_name
                type: (_) @var_type)) @var_def
    """,
    "type_aliases": """
        (type_declaration
            (type_spec
                name: (type_identifier) @type_alias_name
                type: (_) @aliased_type)) @type_alias
    """,
    "goroutines": """
        (go_statement
            (call_expression
                function: (identifier) @goroutine_func)) @goroutine_call
    """,
    "channels": """
        (channel_type
            element: (_) @channel_element_type) @channel_type_def
    """,
    "interface_methods": """
        (interface_type
            (method_spec_list
                (method_spec
                    name: (field_identifier) @interface_method_name
                    parameters: (parameter_list) @interface_method_params) @interface_method)) @interface_with_methods
    """,
    "embedded_fields": """
        (struct_type
            (field_declaration_list
                (field_declaration
                    type: (type_identifier) @embedded_type) @embedded_field)) @struct_with_embedded
    """,
    "pointer_methods": """
        (method_declaration
            receiver: (parameter_list 
                (parameter_declaration
                    type: (pointer_type
                        (_) @pointer_receiver_type))) @pointer_receiver
            name: (identifier) @pointer_method_name) @pointer_method_def
    """,
}

# Go-specific helper patterns
GO_HELPERS = {
    "parameter_patterns": {
        "simple": "(parameter_declaration name: (identifier) @param_name type: (_) @param_type)",
        "variadic": "(variadic_parameter_declaration name: (identifier) @variadic_name type: (_) @variadic_type)",
        "unnamed": "(parameter_declaration type: (_) @unnamed_param_type)",
        "multiple": "(parameter_declaration name: (identifier_list (identifier) @multi_param_name) type: (_) @multi_param_type)",
    },
    "type_patterns": {
        "basic": "(type_identifier) @basic_type",
        "pointer": "(pointer_type (_) @pointed_type)",
        "slice": "(slice_type (_) @slice_element_type)",
        "array": "(array_type length: (_) @array_length element: (_) @array_element_type)",
        "map": "(map_type key: (_) @map_key_type value: (_) @map_value_type)",
        "channel": "(channel_type element: (_) @channel_element_type)",
        "function": "(function_type parameters: (parameter_list) return_type: (_) @func_return_type)",
        "interface": "(interface_type) @interface_type",
        "struct": "(struct_type) @struct_type",
    },
    "receiver_patterns": {
        "value": "(parameter_declaration name: (identifier) @receiver_name type: (type_identifier) @receiver_type)",
        "pointer": "(parameter_declaration name: (identifier) @pointer_receiver_name type: (pointer_type (type_identifier) @pointer_receiver_type))",
        "unnamed_value": "(parameter_declaration type: (type_identifier) @unnamed_receiver_type)",
        "unnamed_pointer": "(parameter_declaration type: (pointer_type (type_identifier) @unnamed_pointer_receiver_type)",
    },
}

# Go language constructs and conventions
GO_CONSTRUCTS = {
    "builtin_types": [
        "bool",
        "string",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
        "byte",
        "rune",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
    "builtin_functions": [
        "make",
        "new",
        "len",
        "cap",
        "append",
        "copy",
        "delete",
        "close",
        "panic",
        "recover",
        "print",
        "println",
        "complex",
        "real",
        "imag",
    ],
    "keywords": [
        "break",
        "case",
        "chan",
        "const",
        "continue",
        "default",
        "defer",
        "else",
        "fallthrough",
        "for",
        "func",
        "go",
        "goto",
        "if",
        "import",
        "interface",
        "map",
        "package",
        "range",
        "return",
        "select",
        "struct",
        "switch",
        "type",
        "var",
    ],
    "common_packages": [
        "fmt",
        "os",
        "io",
        "strings",
        "strconv",
        "time",
        "math",
        "net/http",
        "encoding/json",
        "database/sql",
        "context",
        "sync",
        "log",
        "testing",
        "errors",
        "sort",
        "bufio",
    ],
    "special_methods": [
        "String",
        "Error",
        "GoString",
        "Format",
        "Read",
        "Write",
        "Close",
        "ServeHTTP",
        "Len",
        "Less",
        "Swap",
    ],
    "visibility_rules": {
        "exported": "Starts with uppercase letter",
        "unexported": "Starts with lowercase letter",
    },
}

# Validation patterns for Go identifiers and constructs
VALIDATION_PATTERNS = {
    "valid_identifier": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    "exported_identifier": r"^[A-Z][a-zA-Z0-9_]*$",
    "unexported_identifier": r"^[a-z_][a-zA-Z0-9_]*$",
    "package_name": r"^[a-z][a-z0-9_]*$",
    "constant_name": r"^[A-Z][A-Z0-9_]*$|^[a-z][a-zA-Z0-9_]*$",
}


def get_query(query_type: str) -> str:
    """
    Get a specific Go query pattern by type.

    Args:
        query_type: The type of query to retrieve

    Returns:
        The S-expression query string

    Raises:
        KeyError: If query_type is not found
    """
    if query_type not in GO_QUERIES:
        available = ", ".join(GO_QUERIES.keys())
        raise KeyError(f"Query type '{query_type}' not found. Available: {available}")

    return GO_QUERIES[query_type]


def get_all_queries() -> dict:
    """Get all available Go query patterns."""
    return GO_QUERIES.copy()


def is_builtin_type(type_name: str) -> bool:
    """Check if a type name is a Go builtin type."""
    return type_name in GO_CONSTRUCTS["builtin_types"]


def is_exported(identifier: str) -> bool:
    """Check if an identifier is exported (public) in Go."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["exported_identifier"], identifier))


def is_builtin_function(func_name: str) -> bool:
    """Check if a function name is a Go builtin function."""
    return func_name in GO_CONSTRUCTS["builtin_functions"]


def validate_go_identifier(identifier: str) -> bool:
    """Validate that an identifier follows Go naming rules."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_identifier"], identifier))


def extract_package_from_import(import_path: str) -> str:
    """Extract package name from import path."""
    # Remove quotes if present
    clean_path = import_path.strip("\"'")

    # Get the last part of the path as package name
    if "/" in clean_path:
        return clean_path.split("/")[-1]
    return clean_path


def is_pointer_method(receiver_type: str) -> bool:
    """Check if a method has a pointer receiver."""
    return receiver_type.startswith("*")


def extract_receiver_type(receiver_type: str) -> str:
    """Extract the base type from a receiver type (remove pointer if present)."""
    if receiver_type.startswith("*"):
        return receiver_type[1:]
    return receiver_type
