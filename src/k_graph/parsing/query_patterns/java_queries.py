"""
Java Tree-sitter Query Patterns

S-expression query patterns for extracting code structures from Java source code using Tree-sitter.
These queries handle Java-specific constructs like packages, access modifiers, static methods,
and inheritance while maintaining compatibility with the existing Neo4j schema.
"""

# Query patterns for Java language constructs
JAVA_QUERIES = {
    "classes": """
        (class_declaration
            name: (identifier) @class_name
            body: (class_body) @class_body) @class_def
    """,
    "interfaces": """
        (interface_declaration
            name: (identifier) @interface_name
            body: (interface_body) @interface_body) @interface_def
    """,
    "methods": """
        (class_declaration
            body: (class_body
                (method_declaration
                    name: (identifier) @method_name
                    parameters: (formal_parameters) @method_params
                    body: (block) @method_body) @method_def)) @class_with_methods
    """,
    "constructors": """
        (class_declaration
            name: (identifier) @class_name
            body: (class_body
                (constructor_declaration
                    name: (identifier) @constructor_name
                    parameters: (formal_parameters) @constructor_params
                    body: (constructor_body) @constructor_body) @constructor_def)) @class_with_constructor
    """,
    "static_methods": """
        (class_declaration
            body: (class_body
                (method_declaration
                    (modifiers 
                        (modifier) @static_modifier)
                    name: (identifier) @static_method_name
                    parameters: (formal_parameters) @static_params
                    body: (block) @static_body) @static_method)) @class_with_static_methods
    """,
    "fields": """
        (class_declaration
            body: (class_body
                (field_declaration
                    type: (_) @field_type
                    declarator: (variable_declarator
                        name: (identifier) @field_name)) @field_def)) @class_with_fields
    """,
    "imports": """
        (import_declaration
            (scoped_identifier) @import_name) @import
    """,
    "package_declaration": """
        (package_declaration
            (scoped_identifier) @package_name) @package
    """,
    "inheritance": """
        (class_declaration
            name: (identifier) @class_name
            superclass: (superclass
                (type_identifier) @parent_class)) @inherited_class
    """,
    "interface_implementation": """
        (class_declaration
            name: (identifier) @implementing_class
            interfaces: (super_interfaces
                (interface_type_list
                    (type_identifier) @implemented_interface))) @class_implementing_interface
    """,
    "annotations": """
        (class_declaration
            (modifiers
                (annotation
                    name: (identifier) @annotation_name)) @annotation
            name: (identifier) @annotated_class_name) @annotated_class
    """,
    "method_annotations": """
        (class_declaration
            body: (class_body
                (method_declaration
                    (modifiers
                        (annotation
                            name: (identifier) @method_annotation_name)) @method_annotation
                    name: (identifier) @annotated_method_name) @annotated_method)) @class_with_annotated_methods
    """,
    "enum_declarations": """
        (enum_declaration
            name: (identifier) @enum_name
            body: (enum_body
                (enum_constant
                    name: (identifier) @enum_constant) @enum_const)) @enum_def
    """,
    "generic_classes": """
        (class_declaration
            name: (identifier) @generic_class_name
            type_parameters: (type_parameters
                (type_parameter 
                    name: (type_identifier) @type_param)) @type_params) @generic_class
    """,
    "abstract_classes": """
        (class_declaration
            (modifiers 
                (modifier) @abstract_modifier)
            name: (identifier) @abstract_class_name
            body: (class_body) @abstract_class_body) @abstract_class
    """,
    "inner_classes": """
        (class_declaration
            body: (class_body
                (class_declaration
                    name: (identifier) @inner_class_name
                    body: (class_body) @inner_class_body) @inner_class_def)) @class_with_inner_classes
    """,
}

# Java-specific helper patterns
JAVA_HELPERS = {
    "parameter_patterns": {
        "simple": "(formal_parameter type: (_) @param_type name: (identifier) @param_name)",
        "varargs": "(spread_parameter type: (_) @varargs_type name: (identifier) @varargs_name)",
        "final": "(formal_parameter (modifiers (modifier) @final_modifier) type: (_) @param_type name: (identifier) @param_name)",
    },
    "modifier_patterns": {
        "access": "(modifier) @access_modifier",  # public, private, protected
        "static": "(modifier) @static_modifier",
        "final": "(modifier) @final_modifier",
        "abstract": "(modifier) @abstract_modifier",
        "synchronized": "(modifier) @synchronized_modifier",
        "volatile": "(modifier) @volatile_modifier",
        "transient": "(modifier) @transient_modifier",
        "native": "(modifier) @native_modifier",
    },
    "type_patterns": {
        "primitive": "(integral_type) @primitive_type | (floating_point_type) @primitive_type | (boolean_type) @primitive_type",
        "array": "(array_type element: (_) @element_type)",
        "generic": "(generic_type (type_identifier) @generic_base (type_arguments (_) @type_arg))",
        "wildcard": "(wildcard) @wildcard_type",
        "bounded_wildcard": "(wildcard (wildcard_bounds (_) @bound_type)) @bounded_wildcard",
    },
    "annotation_patterns": {
        "simple": "(annotation name: (identifier) @annotation_name)",
        "with_args": "(annotation name: (identifier) @annotation_name arguments: (annotation_argument_list) @annotation_args)",
        "marker": "(marker_annotation name: (identifier) @marker_name)",
        "single_element": "(single_element_annotation name: (identifier) @single_element_name value: (_) @annotation_value)",
    },
}

# Java language constructs and conventions
JAVA_CONSTRUCTS = {
    "primitive_types": [
        "byte",
        "short",
        "int",
        "long",
        "float",
        "double",
        "boolean",
        "char",
        "void",
    ],
    "wrapper_classes": [
        "Byte",
        "Short",
        "Integer",
        "Long",
        "Float",
        "Double",
        "Boolean",
        "Character",
        "String",
    ],
    "access_modifiers": ["public", "private", "protected"],
    "other_modifiers": [
        "static",
        "final",
        "abstract",
        "synchronized",
        "volatile",
        "transient",
        "native",
        "strictfp",
    ],
    "common_annotations": [
        "Override",
        "Deprecated",
        "SuppressWarnings",
        "FunctionalInterface",
        "SafeVarargs",
        "Generated",
        "PostConstruct",
        "PreDestroy",
        "Inject",
        "Autowired",
        "Component",
        "Service",
        "Repository",
        "Controller",
        "RestController",
        "RequestMapping",
        "GetMapping",
        "PostMapping",
        "PutMapping",
        "DeleteMapping",
    ],
    "builtin_classes": [
        "Object",
        "String",
        "Class",
        "Thread",
        "Runnable",
        "Exception",
        "Error",
        "System",
        "Math",
        "Collections",
        "Arrays",
        "List",
        "Set",
        "Map",
        "Optional",
    ],
    "common_packages": [
        "java.lang",
        "java.util",
        "java.io",
        "java.net",
        "java.time",
        "java.util.concurrent",
        "java.util.stream",
        "java.nio",
        "javax.servlet",
        "org.springframework",
        "org.junit",
    ],
}

# Validation patterns for Java identifiers and constructs
VALIDATION_PATTERNS = {
    "valid_identifier": r"^[a-zA-Z_$][a-zA-Z0-9_$]*$",
    "valid_class_name": r"^[A-Z][a-zA-Z0-9_]*$",
    "valid_method_name": r"^[a-z][a-zA-Z0-9_]*$",
    "valid_constant": r"^[A-Z][A-Z0-9_]*$",
    "valid_package": r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    "generic_param": r"^[A-Z]([A-Z0-9_]*)?$",
}


def get_query(query_type: str) -> str:
    """
    Get a specific Java query pattern by type.

    Args:
        query_type: The type of query to retrieve

    Returns:
        The S-expression query string

    Raises:
        KeyError: If query_type is not found
    """
    if query_type not in JAVA_QUERIES:
        available = ", ".join(JAVA_QUERIES.keys())
        raise KeyError(f"Query type '{query_type}' not found. Available: {available}")

    return JAVA_QUERIES[query_type]


def get_all_queries() -> dict:
    """Get all available Java query patterns."""
    return JAVA_QUERIES.copy()


def is_primitive_type(type_name: str) -> bool:
    """Check if a type name is a Java primitive type."""
    return type_name in JAVA_CONSTRUCTS["primitive_types"]


def is_wrapper_class(type_name: str) -> bool:
    """Check if a type name is a Java wrapper class."""
    return type_name in JAVA_CONSTRUCTS["wrapper_classes"]


def is_access_modifier(modifier: str) -> bool:
    """Check if a modifier is a Java access modifier."""
    return modifier in JAVA_CONSTRUCTS["access_modifiers"]


def validate_java_identifier(identifier: str) -> bool:
    """Validate that an identifier follows Java naming rules."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_identifier"], identifier))


def follows_class_naming(name: str) -> bool:
    """Check if a name follows Java class naming convention."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_class_name"], name))


def follows_method_naming(name: str) -> bool:
    """Check if a name follows Java method naming convention."""
    import re

    return bool(re.match(VALIDATION_PATTERNS["valid_method_name"], name))


def extract_package_from_import(import_statement: str) -> str:
    """Extract package name from an import statement."""
    # Remove 'import ' and ';' if present
    clean_import = import_statement.replace("import ", "").replace(";", "").strip()

    # Split on dots and take all but the last part (class name)
    parts = clean_import.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return ""
