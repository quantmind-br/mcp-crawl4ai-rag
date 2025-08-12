"""
TypeScript Tree-sitter Query Patterns

S-expression query patterns for extracting code structures from TypeScript source code using Tree-sitter.
These queries handle TypeScript-specific constructs like interfaces, type definitions, generics,
and access modifiers while maintaining compatibility with the existing Neo4j schema.
"""

# Query patterns for TypeScript language constructs
TYPESCRIPT_QUERIES = {
    "classes": """
        (class_declaration 
            name: (type_identifier) @class_name
            body: (class_body) @class_body) @class_def
    """,
    "interfaces": """
        (interface_declaration
            name: (type_identifier) @interface_name
            body: (object_type) @interface_body) @interface_def
    """,
    "functions": """
        (function_declaration 
            name: (identifier) @func_name
            parameters: (formal_parameters) @params
            body: (statement_block) @func_body) @func_def
    """,
    "methods": """
        (class_declaration
            body: (class_body
                (method_definition
                    name: (property_identifier) @method_name
                    parameters: (formal_parameters) @method_params
                    body: (statement_block) @method_body) @method_def)) @class_with_methods
    """,
    "arrow_functions": """
        (variable_declaration
            (variable_declarator
                name: (identifier) @arrow_func_name
                value: (arrow_function
                    parameters: (formal_parameters) @arrow_params
                    body: (_) @arrow_body))) @arrow_func_decl
    """,
    "imports": """
        (import_statement
            source: (string) @import_source) @import
    """,
    "named_imports": """
        (import_statement
            (import_clause
                (named_imports
                    (import_specifier
                        name: (identifier) @import_name))) @named_import_clause
            source: (string) @import_source) @named_import
    """,
    "default_imports": """
        (import_statement
            (import_clause
                (identifier) @default_import_name)
            source: (string) @import_source) @default_import
    """,
    "type_definitions": """
        (type_alias_declaration
            name: (type_identifier) @type_name
            value: (_) @type_value) @type_def
    """,
    "class_properties": """
        (class_declaration
            body: (class_body
                (property_definition
                    name: (property_identifier) @prop_name
                    type: (_) @prop_type) @prop_def)) @class_with_props
    """,
    "interface_properties": """
        (interface_declaration
            body: (object_type
                (property_signature
                    name: (property_identifier) @interface_prop_name
                    type: (_) @interface_prop_type) @interface_prop)) @interface_with_props
    """,
    "generics": """
        (class_declaration
            name: (type_identifier) @generic_class_name
            type_parameters: (type_parameters
                (type_parameter 
                    name: (type_identifier) @type_param)) @type_params) @generic_class
    """,
    "decorators": """
        (class_declaration
            (decorator
                (identifier) @decorator_name) @decorator
            name: (type_identifier) @decorated_class_name) @decorated_class
    """,
    "access_modifiers": """
        (class_declaration
            body: (class_body
                (method_definition
                    (accessibility_modifier) @access_modifier
                    name: (property_identifier) @modified_method_name) @modified_method)) @class_with_access_modifiers
    """,
    "export_declarations": """
        (export_statement
            declaration: (_) @exported_item) @export
    """,
    "namespace_declarations": """
        (namespace_declaration
            name: (identifier) @namespace_name
            body: (statement_block) @namespace_body) @namespace_def
    """,
}

# TypeScript-specific helper patterns
TYPESCRIPT_HELPERS = {
    "parameter_patterns": {
        "simple": "(identifier) @param_name",
        "typed": "(required_parameter (identifier) @param_name (type_annotation (_) @param_type))",
        "optional": "(optional_parameter (identifier) @param_name)",
        "default": "(assignment_pattern (identifier) @param_name (_) @default_value)",
        "rest": "(rest_parameter (identifier) @rest_param_name)",
    },
    "type_patterns": {
        "primitive": "(predefined_type) @primitive_type",
        "identifier": "(type_identifier) @type_id",
        "generic": "(generic_type (type_identifier) @generic_base (type_arguments (_) @type_arg))",
        "union": "(union_type (_) @union_member)",
        "intersection": "(intersection_type (_) @intersection_member)",
        "array": "(array_type (_) @array_element_type)",
        "tuple": "(tuple_type (_) @tuple_element)",
        "function_type": "(function_type parameters: (formal_parameters) return_type: (_) @return_type)",
        "object_type": "(object_type (property_signature) @object_prop)",
    },
    "modifier_patterns": {
        "access": "(accessibility_modifier) @access",  # public, private, protected
        "static": "(static) @static_modifier",
        "readonly": "(readonly) @readonly_modifier",
        "abstract": "(abstract) @abstract_modifier",
        "async": "(async) @async_modifier",
    },
}

# TypeScript language constructs and conventions
TYPESCRIPT_CONSTRUCTS = {
    "primitive_types": [
        "string",
        "number",
        "boolean",
        "object",
        "undefined",
        "null",
        "void",
        "never",
        "any",
        "unknown",
        "bigint",
        "symbol",
    ],
    "utility_types": [
        "Partial",
        "Required",
        "Readonly",
        "Pick",
        "Omit",
        "Exclude",
        "Extract",
        "NonNullable",
        "Parameters",
        "ReturnType",
        "Record",
    ],
    "access_modifiers": ["public", "private", "protected"],
    "common_decorators": [
        "Component",
        "Injectable",
        "Directive",
        "Pipe",
        "Input",
        "Output",
        "ViewChild",
        "ViewChildren",
        "ContentChild",
        "ContentChildren",
    ],
    "builtin_interfaces": [
        "Array",
        "Object",
        "Function",
        "String",
        "Number",
        "Boolean",
        "Date",
        "RegExp",
        "Error",
        "Promise",
        "Map",
        "Set",
        "WeakMap",
        "WeakSet",
    ],
    "common_imports": [
        "react",
        "vue",
        "angular",
        "express",
        "lodash",
        "axios",
        "moment",
        "rxjs",
        "typescript",
        "jest",
        "@types/node",
    ],
}

# Validation patterns for TypeScript identifiers and constructs
VALIDATION_PATTERNS = {
    "valid_identifier": r"^[a-zA-Z_$][a-zA-Z0-9_$]*$",
    "valid_type_name": r"^[A-Z][a-zA-Z0-9_]*$",
    "valid_interface": r"^I[A-Z][a-zA-Z0-9_]*$",  # Common convention
    "private_member": r"^_[a-zA-Z0-9_$]*$",
    "constant": r"^[A-Z][A-Z0-9_]*$",
    "generic_param": r"^T[A-Z]?[a-zA-Z0-9_]*$|^[A-Z]$",
}


def get_query(query_type: str) -> str:
    """
    Get a specific TypeScript query pattern by type.

    Args:
        query_type: The type of query to retrieve

    Returns:
        The S-expression query string

    Raises:
        KeyError: If query_type is not found
    """
    if query_type not in TYPESCRIPT_QUERIES:
        available = ", ".join(TYPESCRIPT_QUERIES.keys())
        raise KeyError(f"Query type '{query_type}' not found. Available: {available}")

    return TYPESCRIPT_QUERIES[query_type]


def get_all_queries() -> dict:
    """Get all available TypeScript query patterns."""
    return TYPESCRIPT_QUERIES.copy()


def is_primitive_type(type_name: str) -> bool:
    """Check if a type name is a TypeScript primitive type."""
    return type_name in TYPESCRIPT_CONSTRUCTS["primitive_types"]


def is_utility_type(type_name: str) -> bool:
    """Check if a type name is a TypeScript utility type."""
    return type_name in TYPESCRIPT_CONSTRUCTS["utility_types"]


def is_interface_convention(name: str) -> bool:
    """Check if a name follows TypeScript interface naming convention."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_interface"], name))


def validate_typescript_identifier(identifier: str) -> bool:
    """Validate that an identifier follows TypeScript naming rules."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_identifier"], identifier))


def extract_generic_parameters(type_string: str) -> list:
    """Extract generic parameter names from a type string."""
    import re

    # Simple regex to extract generic params - could be enhanced
    matches = re.findall(r"<([^>]+)>", type_string)
    if matches:
        return [param.strip() for param in matches[0].split(",")]
    return []
