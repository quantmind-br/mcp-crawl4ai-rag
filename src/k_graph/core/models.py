"""Core data models for knowledge graph components.

This module contains all shared data classes used across the knowledge graph
system, promoting consistency and reusability.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# Validation enums
class ValidationStatus(Enum):
    """Status enumeration for validation results."""

    VALID = "VALID"
    INVALID = "INVALID"
    UNCERTAIN = "UNCERTAIN"
    NOT_FOUND = "NOT_FOUND"


# Parsing models
@dataclass
class ParsedFunction:
    """Represents a parsed function from any language."""

    name: str
    full_name: str
    params: List[str]
    return_type: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None


@dataclass
class ParsedMethod:
    """Represents a parsed method from any language."""

    name: str
    params: List[str]
    return_type: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None


@dataclass
class ParsedAttribute:
    """Represents a parsed attribute/field from any language."""

    name: str
    type: str
    line_start: int
    line_end: int


@dataclass
class ParsedClass:
    """Represents a parsed class from any language."""

    name: str
    full_name: str
    methods: List[ParsedMethod]
    attributes: List[ParsedAttribute]
    line_start: int
    line_end: int
    docstring: Optional[str] = None


@dataclass
class ParseResult:
    """Standard result structure that matches existing AST parser output."""

    module_name: str
    file_path: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[str]
    line_count: int
    language: str
    errors: List[str]


# Analysis models
@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    name: str
    alias: Optional[str] = None
    is_from_import: bool = False
    line_number: int = 0


@dataclass
class MethodCall:
    """Information about a method call."""

    object_name: str
    method_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    object_type: Optional[str] = None  # Inferred class type


@dataclass
class AttributeAccess:
    """Information about attribute access."""

    object_name: str
    attribute_name: str
    line_number: int
    object_type: Optional[str] = None  # Inferred class type


@dataclass
class FunctionCall:
    """Information about a function call."""

    function_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    full_name: Optional[str] = None  # Module.function_name


@dataclass
class ClassInstantiation:
    """Information about class instantiation."""

    variable_name: str
    class_name: str
    args: List[str]
    kwargs: Dict[str, str]
    line_number: int
    full_class_name: Optional[str] = None  # Module.ClassName


@dataclass
class AnalysisResult:
    """Complete analysis results for a multi-language script."""

    file_path: str
    language: str = "unknown"  # Programming language detected
    imports: List[ImportInfo] = field(default_factory=list)
    class_instantiations: List[ClassInstantiation] = field(default_factory=list)
    method_calls: List[MethodCall] = field(default_factory=list)
    attribute_accesses: List[AttributeAccess] = field(default_factory=list)
    function_calls: List[FunctionCall] = field(default_factory=list)
    variable_types: Dict[str, str] = field(
        default_factory=dict
    )  # variable_name -> class_type
    errors: List[str] = field(default_factory=list)

    # Additional Tree-sitter data for cross-validation
    tree_sitter_data: Optional[ParseResult] = field(default=None)


# Validation models
@dataclass
class ValidationResult:
    """Result of validating a single element."""

    status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ImportValidation:
    """Validation result for an import."""

    import_info: ImportInfo
    validation: ValidationResult
    available_classes: List[str] = field(default_factory=list)
    available_functions: List[str] = field(default_factory=list)


@dataclass
class MethodValidation:
    """Validation result for a method call."""

    method_call: MethodCall
    validation: ValidationResult
    expected_params: List[str] = field(default_factory=list)
    actual_params: List[str] = field(default_factory=list)
    parameter_validation: Optional[ValidationResult] = None


@dataclass
class AttributeValidation:
    """Validation result for attribute access."""

    attribute_access: AttributeAccess
    validation: ValidationResult
    expected_type: Optional[str] = None


@dataclass
class FunctionValidation:
    """Validation result for function call."""

    function_call: FunctionCall
    validation: ValidationResult
    expected_params: List[str] = field(default_factory=list)
    actual_params: List[str] = field(default_factory=list)
    parameter_validation: Optional[ValidationResult] = None


@dataclass
class ClassValidation:
    """Validation result for class instantiation."""

    class_instantiation: ClassInstantiation
    validation: ValidationResult
    constructor_params: List[str] = field(default_factory=list)
    parameter_validation: Optional[ValidationResult] = None


@dataclass
class ScriptValidationResult:
    """Complete validation results for a script."""

    script_path: str
    analysis_result: AnalysisResult
    import_validations: List[ImportValidation] = field(default_factory=list)
    class_validations: List[ClassValidation] = field(default_factory=list)
    method_validations: List[MethodValidation] = field(default_factory=list)
    function_validations: List[FunctionValidation] = field(default_factory=list)
    attribute_validations: List[AttributeValidation] = field(default_factory=list)
    overall_confidence: float = 0.0
    is_likely_hallucinated: bool = False
    hallucination_indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
