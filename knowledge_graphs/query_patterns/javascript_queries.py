"""
JavaScript Tree-sitter query patterns.

This module contains S-expression query patterns for extracting structural information
from JavaScript source code using Tree-sitter. These patterns target classes, functions,
methods, imports, and other language constructs.
"""


def get_all_queries() -> dict:
    """Get all available JavaScript query patterns."""
    return JAVASCRIPT_QUERIES


def get_query(query_type: str) -> str:
    """Get a specific JavaScript query pattern."""
    if query_type not in JAVASCRIPT_QUERIES:
        available = ", ".join(JAVASCRIPT_QUERIES.keys())
        raise KeyError(
            f"Query type '{query_type}' not available. Available: {available}"
        )
    return JAVASCRIPT_QUERIES[query_type]


JAVASCRIPT_QUERIES = {
    "classes": """
    (class_declaration
      name: (identifier) @class_name) @class_definition
    """,
    "functions": """
    (function_declaration
      name: (identifier) @func_name
      parameters: (formal_parameters) @func_parameters) @func_definition
      
    (variable_declarator
      name: (identifier) @func_name
      value: (arrow_function) @func_definition)
      
    (assignment_expression
      left: (identifier) @func_name
      right: (arrow_function) @func_definition)
    """,
    "methods": """
    (class_body
      (method_definition
        name: (property_name) @method_name
        value: (function) @method_definition))
        
    (class_body
      (method_definition
        name: (property_name) @method_name
        parameters: (formal_parameters) @method_parameters) @method_definition)
    """,
    "imports": """
    (import_statement
      source: (string) @import_source) @import_definition
    
    (import_statement
      (import_clause
        (named_imports
          (import_specifier
            name: (identifier) @import_name))) @import_definition
      source: (string) @import_source)
      
    (import_statement
      (import_clause
        (namespace_import
          (identifier) @import_name)) @import_definition
      source: (string) @import_source)
      
    (import_statement
      (import_clause
        (identifier) @import_name) @import_definition
      source: (string) @import_source)
    
    (call_expression
      function: (identifier) @import_function
      arguments: (arguments (string) @import_source)) @import_definition
      (#eq? @import_function "require")
    """,
}
