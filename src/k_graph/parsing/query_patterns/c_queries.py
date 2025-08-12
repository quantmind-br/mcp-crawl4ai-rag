"""
C Tree-sitter Query Patterns

S-expression query patterns for extracting code structures from C source code using Tree-sitter.
These queries handle C-specific constructs like function declarations, struct definitions,
preprocessor directives, and pointer declarations.
"""

# Query patterns for C language constructs
C_QUERIES = {
    "functions": """
        (function_definition
            type: (_) @func_return_type
            declarator: (function_declarator
                declarator: (identifier) @func_name
                parameters: (parameter_list) @func_params)
            body: (compound_statement) @func_body) @func_def
    """,
    "function_declarations": """
        (declaration
            type: (_) @decl_return_type
            declarator: (function_declarator
                declarator: (identifier) @decl_func_name
                parameters: (parameter_list) @decl_params)) @func_declaration
    """,
    "structs": """
        (struct_specifier
            name: (type_identifier) @struct_name
            body: (field_declaration_list) @struct_body) @struct_def
    """,
    "unions": """
        (union_specifier
            name: (type_identifier) @union_name
            body: (field_declaration_list) @union_body) @union_def
    """,
    "enums": """
        (enum_specifier
            name: (type_identifier) @enum_name
            body: (enumerator_list
                (enumerator
                    name: (identifier) @enum_value) @enum_item)) @enum_def
    """,
    "typedefs": """
        (type_definition
            type: (_) @typedef_type
            declarator: (type_identifier) @typedef_name) @typedef_def
    """,
    "global_variables": """
        (declaration
            type: (_) @var_type
            declarator: (identifier) @var_name) @global_var
    """,
    "struct_fields": """
        (struct_specifier
            body: (field_declaration_list
                (field_declaration
                    type: (_) @field_type
                    declarator: (field_identifier) @field_name) @field_decl)) @struct_with_fields
    """,
    "includes": """
        (preproc_include
            path: (system_lib_string) @system_include) @system_include_directive
    """,
    "local_includes": """
        (preproc_include
            path: (string_literal) @local_include) @local_include_directive  
    """,
    "defines": """
        (preproc_def
            name: (identifier) @define_name
            value: (_) @define_value) @define_directive
    """,
    "function_like_macros": """
        (preproc_function_def
            name: (identifier) @macro_name
            parameters: (preproc_params) @macro_params
            value: (_) @macro_body) @macro_def
    """,
    "static_functions": """
        (function_definition
            (storage_class_specifier) @static_specifier
            type: (_) @static_func_return_type
            declarator: (function_declarator
                declarator: (identifier) @static_func_name) @static_func_declarator) @static_func_def
    """,
    "pointer_declarations": """
        (declaration
            type: (_) @pointer_base_type
            declarator: (pointer_declarator
                (pointer_declarator
                    declarator: (identifier) @pointer_name))) @pointer_decl
    """,
    "array_declarations": """
        (declaration
            type: (_) @array_base_type
            declarator: (array_declarator
                declarator: (identifier) @array_name
                size: (_) @array_size)) @array_decl
    """,
    "function_pointers": """
        (declaration
            type: (_) @func_ptr_return_type
            declarator: (function_declarator
                declarator: (parenthesized_declarator
                    (pointer_declarator
                        declarator: (identifier) @func_ptr_name))
                parameters: (parameter_list) @func_ptr_params)) @func_ptr_decl
    """,
}

# C-specific helper patterns
C_HELPERS = {
    "parameter_patterns": {
        "simple": "(parameter_declaration type: (_) @param_type declarator: (identifier) @param_name)",
        "pointer": "(parameter_declaration type: (_) @pointer_param_type declarator: (pointer_declarator declarator: (identifier) @pointer_param_name))",
        "array": "(parameter_declaration type: (_) @array_param_type declarator: (array_declarator declarator: (identifier) @array_param_name))",
        "void": "(parameter_declaration type: (primitive_type) @void_param)",
    },
    "type_patterns": {
        "primitive": "(primitive_type) @primitive_type",
        "struct": "(struct_specifier name: (type_identifier) @struct_type)",
        "union": "(union_specifier name: (type_identifier) @union_type)",
        "enum": "(enum_specifier name: (type_identifier) @enum_type)",
        "typedef": "(type_identifier) @typedef_type",
        "pointer": "(pointer_declarator (_) @pointed_type)",
        "array": "(array_declarator declarator: (_) @array_element size: (_) @array_size)",
    },
    "storage_class_patterns": {
        "static": "(storage_class_specifier) @static",
        "extern": "(storage_class_specifier) @extern",
        "auto": "(storage_class_specifier) @auto",
        "register": "(storage_class_specifier) @register",
    },
    "type_qualifier_patterns": {
        "const": "(type_qualifier) @const",
        "volatile": "(type_qualifier) @volatile",
        "restrict": "(type_qualifier) @restrict",
    },
}

# C language constructs and conventions
C_CONSTRUCTS = {
    "primitive_types": [
        "void",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "signed",
        "unsigned",
        "_Bool",
        "_Complex",
        "_Imaginary",
    ],
    "storage_classes": ["auto", "register", "static", "extern", "typedef"],
    "type_qualifiers": ["const", "volatile", "restrict", "inline"],
    "standard_headers": [
        "stdio.h",
        "stdlib.h",
        "string.h",
        "math.h",
        "time.h",
        "ctype.h",
        "stdarg.h",
        "setjmp.h",
        "signal.h",
        "assert.h",
        "errno.h",
        "limits.h",
        "float.h",
        "stddef.h",
        "locale.h",
    ],
    "common_functions": [
        "printf",
        "scanf",
        "malloc",
        "free",
        "memcpy",
        "memset",
        "strcpy",
        "strcat",
        "strlen",
        "strcmp",
        "fopen",
        "fclose",
        "fread",
        "fwrite",
        "exit",
        "atoi",
        "atof",
    ],
    "naming_conventions": {
        "constants": "UPPER_SNAKE_CASE",
        "macros": "UPPER_SNAKE_CASE",
        "functions": "snake_case or camelCase",
        "variables": "snake_case or camelCase",
        "types": "snake_case_t or CamelCase",
    },
}

# Validation patterns for C identifiers and constructs
VALIDATION_PATTERNS = {
    "valid_identifier": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    "macro_constant": r"^[A-Z][A-Z0-9_]*$",
    "function_name": r"^[a-z_][a-zA-Z0-9_]*$",
    "type_name": r"^[a-z_][a-zA-Z0-9_]*_t$|^[A-Z][a-zA-Z0-9_]*$",
    "header_guard": r"^[A-Z][A-Z0-9_]*_H$",
}


def get_query(query_type: str) -> str:
    """
    Get a specific C query pattern by type.

    Args:
        query_type: The type of query to retrieve

    Returns:
        The S-expression query string

    Raises:
        KeyError: If query_type is not found
    """
    if query_type not in C_QUERIES:
        available = ", ".join(C_QUERIES.keys())
        raise KeyError(f"Query type '{query_type}' not found. Available: {available}")

    return C_QUERIES[query_type]


def get_all_queries() -> dict:
    """Get all available C query patterns."""
    return C_QUERIES.copy()


def is_primitive_type(type_name: str) -> bool:
    """Check if a type name is a C primitive type."""
    return type_name in C_CONSTRUCTS["primitive_types"]


def is_standard_header(header_name: str) -> bool:
    """Check if a header name is a C standard library header."""
    return header_name in C_CONSTRUCTS["standard_headers"]


def validate_c_identifier(identifier: str) -> bool:
    """Validate that an identifier follows C naming rules."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_identifier"], identifier))


def is_macro_style(name: str) -> bool:
    """Check if a name follows C macro/constant naming convention."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["macro_constant"], name))


def extract_include_path(include_directive: str) -> str:
    """Extract file path from include directive."""
    # Remove #include and whitespace
    clean = include_directive.replace("#include", "").strip()

    # Remove < > or " " brackets
    if clean.startswith("<") and clean.endswith(">"):
        return clean[1:-1]
    elif clean.startswith('"') and clean.endswith('"'):
        return clean[1:-1]

    return clean


def is_function_like_macro(macro_def: str) -> bool:
    """Check if a macro definition is function-like."""
    return "(" in macro_def and ")" in macro_def
