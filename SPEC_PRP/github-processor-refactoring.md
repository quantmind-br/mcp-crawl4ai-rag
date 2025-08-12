# GitHub Processor Refactoring PRP

**Specification**: Refatore o arquivo `.\src\features\github_processor.py` seguindo as melhores práticas visando um código mais modular e de fácil manutenção. Garanta que o código permaneça totalmente funcional.

## Current State Assessment

### File Analysis
```yaml
current_state:
  files: 
    - src/features/github_processor.py (1394 lines, 10 classes)
  behavior: 
    - Monolithic file with multiple responsibilities
    - Tightly coupled classes with mixed concerns
    - Hard-coded dependencies and configurations
    - Complex inheritance and composition patterns
  issues:
    - Single file contains 1394 lines (violates CLAUDE.md 500-line limit)
    - Mixed abstraction levels (low-level file ops + high-level orchestration)
    - Tight coupling between components
    - Inconsistent error handling patterns
    - Repeated code patterns across processors
    - Hard to test individual components
    - Difficult to extend with new file types
```

### Technical Debt Identified

1. **Monolithic Structure**: Single 1394-line file with 10 classes
2. **Mixed Responsibilities**: File processing, Git operations, metadata extraction all in one file
3. **Tight Coupling**: Classes depend directly on each other's implementation details
4. **Code Duplication**: Similar patterns repeated across file processors
5. **Hard-coded Constants**: File size limits, excluded patterns scattered throughout
6. **Inconsistent Error Handling**: Different exception patterns across classes
7. **Testing Complexity**: Hard to test individual components in isolation

### Integration Points
- **Used by**: `src/tools/github_tools.py`, `src/services/unified_indexing_service.py`
- **Imports**: Standard library, typing, subprocess, ast, logging, re
- **External Dependencies**: None (self-contained)
- **Test Coverage**: `tests/features/test_github_processor.py`, `tests/github/test_github_processor.py`

## Desired State Research

### Best Practices for Modular Design

#### 1. **Separation of Concerns (SoC)**
- Each module has a single, well-defined responsibility
- Clear boundaries between different types of operations
- Loose coupling through dependency injection

#### 2. **SOLID Principles**
- **S**ingle Responsibility: Each class handles one concern
- **O**pen/Closed: Open for extension, closed for modification
- **L**iskov Substitution: Subclasses must be substitutable
- **I**nterface Segregation: Many client-specific interfaces
- **D**ependency Inversion: Depend on abstractions, not concretions

#### 3. **Clean Architecture Patterns**
- **Repository Pattern**: Abstract data access
- **Factory Pattern**: Create objects without specifying exact class
- **Strategy Pattern**: Encapsulate algorithms and make them interchangeable
- **Command Pattern**: Encapsulate operations as objects

#### 4. **Modular Structure**
- **Core Domain**: Business logic and entities
- **Use Cases**: Application-specific business rules
- **Interface Adapters**: Controllers, presenters, gateways
- **Frameworks & Drivers**: External concerns (file system, subprocess)

## Desired State Definition

```yaml
desired_state:
  files:
    - src/features/github/
        - __init__.py
        - core/
            - interfaces.py (Abstract base classes)
            - models.py (Data classes and DTOs)
            - exceptions.py (Custom exceptions)
        - repository/
            - git_operations.py (Git cloning and management)
            - metadata_extractor.py (Repository metadata)
        - discovery/
            - file_discovery.py (File discovery engine)
            - filters.py (File filtering rules)
        - processors/
            - __init__.py
            - base_processor.py (Abstract processor)
            - markdown_processor.py
            - python_processor.py
            - typescript_processor.py
            - config_processor.py
            - processor_factory.py (Factory for processors)
        - services/
            - github_service.py (Main orchestrator)
        - config/
            - settings.py (Configuration constants)
  behavior:
    - Clear separation between Git operations, file discovery, and processing
    - Pluggable processors that can be easily extended
    - Dependency injection for loose coupling
    - Consistent error handling throughout
    - Configuration-driven behavior
  benefits:
    - Easier to test (isolated components)
    - Easier to maintain (single responsibility)
    - Easier to extend (new processors)
    - Better code reuse across project
    - Improved readability and documentation
```

## Hierarchical Objectives

### 1. High-Level: Code Modularization
**Goal**: Transform monolithic file into well-structured, modular architecture following Clean Architecture principles

**Success Criteria**:
- [ ] No single file exceeds 500 lines (CLAUDE.md compliance)
- [ ] Clear separation of concerns across modules
- [ ] All existing functionality preserved
- [ ] All tests continue to pass
- [ ] No breaking changes to public API

### 2. Mid-Level: Component Extraction

#### 2.1. Extract Core Interfaces and Models
**Goal**: Define clear contracts and data structures

**Tasks**:
- Create abstract base classes for processors
- Define data transfer objects (DTOs)
- Create custom exception hierarchy

#### 2.2. Extract Repository Operations  
**Goal**: Isolate Git and file system operations

**Tasks**:
- Extract `GitHubRepoManager` to dedicated module
- Extract `GitHubMetadataExtractor` to dedicated module
- Create repository interface for dependency injection

#### 2.3. Extract File Discovery Engine
**Goal**: Separate file discovery from processing

**Tasks**:
- Extract `MarkdownDiscovery` and `MultiFileDiscovery`
- Create configurable filtering system
- Implement discovery interface

#### 2.4. Extract and Refactor Processors
**Goal**: Create pluggable processor system

**Tasks**:
- Create base processor abstraction
- Extract individual processors to separate files
- Implement processor factory
- Add processor registration system

#### 2.5. Create Service Layer
**Goal**: Orchestrate operations through service layer

**Tasks**:
- Create main `GitHubService` orchestrator
- Implement dependency injection
- Add configuration management

### 3. Low-Level: Implementation Tasks

## Implementation Strategy

### Phase 1: Foundation (Setup and Interfaces)
```yaml
phase1_foundation:
  priority: 1
  dependencies: []
  tasks:
    - create_directory_structure
    - define_interfaces
    - create_data_models
    - setup_exception_hierarchy
    - extract_configuration
```

### Phase 2: Repository Layer (Git and File Operations)
```yaml
phase2_repository:
  priority: 2
  dependencies: [phase1_foundation]
  tasks:
    - extract_git_operations
    - extract_metadata_extraction
    - create_repository_interfaces
```

### Phase 3: Discovery Layer (File Discovery)
```yaml
phase3_discovery:
  priority: 3
  dependencies: [phase1_foundation, phase2_repository]
  tasks:
    - extract_file_discovery
    - create_filtering_system
    - implement_discovery_interfaces
```

### Phase 4: Processor Layer (File Processing)
```yaml
phase4_processors:
  priority: 4
  dependencies: [phase1_foundation]
  tasks:
    - create_base_processor
    - extract_individual_processors
    - implement_processor_factory
    - setup_processor_registry
```

### Phase 5: Service Layer (Orchestration)
```yaml
phase5_services:
  priority: 5
  dependencies: [phase2_repository, phase3_discovery, phase4_processors]
  tasks:
    - create_main_service
    - implement_dependency_injection
    - integrate_all_components
```

### Phase 6: Integration (Testing and Cleanup)
```yaml
phase6_integration:
  priority: 6
  dependencies: [phase5_services]
  tasks:
    - update_imports
    - run_tests
    - verify_functionality
    - cleanup_old_file
```

## Detailed Task Specification

### Task 1: Create Directory Structure
```yaml
create_directory_structure:
  action: CREATE
  files:
    - src/features/github/__init__.py
    - src/features/github/core/__init__.py
    - src/features/github/repository/__init__.py
    - src/features/github/discovery/__init__.py
    - src/features/github/processors/__init__.py
    - src/features/github/services/__init__.py
    - src/features/github/config/__init__.py
  changes: |
    Create modular directory structure with:
    - core/ for interfaces, models, exceptions
    - repository/ for Git and metadata operations
    - discovery/ for file discovery
    - processors/ for file processing
    - services/ for orchestration
    - config/ for configuration
  validation:
    - command: "ls -la src/features/github/"
    - expect: "All directories created successfully"
```

### Task 2: Define Core Interfaces
```yaml
define_interfaces:
  action: CREATE
  file: src/features/github/core/interfaces.py
  changes: |
    Create abstract base classes:
    - IGitRepository: Git operations interface
    - IFileDiscovery: File discovery interface  
    - IFileProcessor: File processor interface
    - IMetadataExtractor: Metadata extraction interface
    - IProcessorFactory: Processor factory interface
    Using Python's ABC (Abstract Base Class) module
  validation:
    - command: "python -c \"from src.features.github.core.interfaces import IGitRepository; print('Interfaces created')\""
    - expect: "Interfaces created"
```

### Task 3: Create Data Models
```yaml
create_data_models:
  action: CREATE
  file: src/features/github/core/models.py
  changes: |
    Create Pydantic models for:
    - RepositoryInfo: Repository metadata
    - FileInfo: File metadata
    - ProcessingResult: Processing results
    - CloneResult: Clone operation results
    Using dataclasses or Pydantic for validation
  validation:
    - command: "python -c \"from src.features.github.core.models import RepositoryInfo; print('Models created')\""
    - expect: "Models created"
```

### Task 4: Setup Exception Hierarchy
```yaml
setup_exception_hierarchy:
  action: CREATE
  file: src/features/github/core/exceptions.py
  changes: |
    Create custom exceptions:
    - GitHubProcessorError: Base exception
    - CloneError: Repository cloning errors
    - DiscoveryError: File discovery errors
    - ProcessingError: File processing errors
    - ValidationError: Data validation errors
    All inheriting from GitHubProcessorError
  validation:
    - command: "python -c \"from src.features.github.core.exceptions import CloneError; print('Exceptions created')\""
    - expect: "Exceptions created"
```

### Task 5: Extract Configuration
```yaml
extract_configuration:
  action: CREATE  
  file: src/features/github/config/settings.py
  changes: |
    Extract all constants to configuration:
    - EXCLUDED_DIRS: Directory exclusion list
    - EXCLUDED_PATTERNS: File pattern exclusions
    - FILE_SIZE_LIMITS: Size limits by file type
    - SUPPORTED_EXTENSIONS: Supported file extensions
    - DEFAULT_LIMITS: Default processing limits
    Use dataclass or Pydantic for configuration
  validation:
    - command: "python -c \"from src.features.github.config.settings import EXCLUDED_DIRS; print(len(EXCLUDED_DIRS))\""
    - expect: "16"
```

### Task 6: Extract Git Operations  
```yaml
extract_git_operations:
  action: CREATE
  file: src/features/github/repository/git_operations.py
  changes: |
    Extract GitHubRepoManager class:
    - Implement IGitRepository interface
    - Move all Git-related operations
    - Add dependency injection for configuration
    - Improve error handling with custom exceptions
    - Add comprehensive logging
  validation:
    - command: "python -c \"from src.features.github.repository.git_operations import GitRepository; r=GitRepository(); print('Git operations extracted')\""
    - expect: "Git operations extracted"
```

### Task 7: Extract Metadata Extraction
```yaml
extract_metadata_extraction:
  action: CREATE
  file: src/features/github/repository/metadata_extractor.py
  changes: |
    Extract GitHubMetadataExtractor class:
    - Implement IMetadataExtractor interface
    - Move all metadata extraction logic
    - Add configuration injection
    - Improve error handling
    - Add better typing
  validation:
    - command: "python -c \"from src.features.github.repository.metadata_extractor import MetadataExtractor; print('Metadata extracted')\""
    - expect: "Metadata extracted"
```

### Task 8: Extract File Discovery
```yaml
extract_file_discovery:
  action: CREATE
  file: src/features/github/discovery/file_discovery.py
  changes: |
    Extract MarkdownDiscovery and MultiFileDiscovery:
    - Implement IFileDiscovery interface
    - Unify discovery logic
    - Add configurable filtering
    - Improve type annotations
    - Add comprehensive error handling
  validation:
    - command: "python -c \"from src.features.github.discovery.file_discovery import FileDiscovery; print('Discovery extracted')\""
    - expect: "Discovery extracted"
```

### Task 9: Create Filtering System
```yaml
create_filtering_system:
  action: CREATE
  file: src/features/github/discovery/filters.py
  changes: |
    Create configurable filtering system:
    - DirectoryFilter: Filter directories
    - FileFilter: Filter files by pattern/size
    - CompositeFilter: Combine multiple filters
    - FilterChain: Process files through filter chain
  validation:
    - command: "python -c \"from src.features.github.discovery.filters import FileFilter; print('Filters created')\""
    - expect: "Filters created"
```

### Task 10: Create Base Processor
```yaml
create_base_processor:
  action: CREATE
  file: src/features/github/processors/base_processor.py
  changes: |
    Create abstract base processor:
    - Implement IFileProcessor interface
    - Define common processing methods
    - Add validation and error handling
    - Provide template method pattern
  validation:
    - command: "python -c \"from src.features.github.processors.base_processor import BaseProcessor; print('Base processor created')\""
    - expect: "Base processor created"
```

### Task 11: Extract Markdown Processor
```yaml
extract_markdown_processor:
  action: CREATE
  file: src/features/github/processors/markdown_processor.py
  changes: |
    Extract MarkdownProcessor:
    - Inherit from BaseProcessor
    - Move markdown processing logic
    - Add configuration injection
    - Improve error handling
  validation:
    - command: "python -c \"from src.features.github.processors.markdown_processor import MarkdownProcessor; print('Markdown processor extracted')\""
    - expect: "Markdown processor extracted"
```

### Task 12: Extract Python Processor
```yaml
extract_python_processor:
  action: CREATE
  file: src/features/github/processors/python_processor.py
  changes: |
    Extract PythonProcessor:
    - Inherit from BaseProcessor
    - Move Python AST processing logic
    - Add better error handling for syntax errors
    - Improve type annotations
  validation:
    - command: "python -c \"from src.features.github.processors.python_processor import PythonProcessor; print('Python processor extracted')\""
    - expect: "Python processor extracted"
```

### Task 13: Extract TypeScript Processor
```yaml
extract_typescript_processor:
  action: CREATE
  file: src/features/github/processors/typescript_processor.py
  changes: |
    Extract TypeScriptProcessor:
    - Inherit from BaseProcessor
    - Move TypeScript JSDoc processing
    - Improve regex patterns
    - Add better error handling
  validation:
    - command: "python -c \"from src.features.github.processors.typescript_processor import TypeScriptProcessor; print('TypeScript processor extracted')\""
    - expect: "TypeScript processor extracted"
```

### Task 14: Extract Config Processor
```yaml
extract_config_processor:
  action: CREATE
  file: src/features/github/processors/config_processor.py
  changes: |
    Extract ConfigProcessor:
    - Inherit from BaseProcessor
    - Move configuration file processing
    - Add support for more config formats
    - Improve language detection
  validation:
    - command: "python -c \"from src.features.github.processors.config_processor import ConfigProcessor; print('Config processor extracted')\""
    - expect: "Config processor extracted"
```

### Task 15: Create Processor Factory
```yaml
create_processor_factory:
  action: CREATE
  file: src/features/github/processors/processor_factory.py
  changes: |
    Create processor factory:
    - Implement IProcessorFactory interface
    - Register processors by file extension
    - Support dynamic processor loading
    - Add processor configuration
  validation:
    - command: "python -c \"from src.features.github.processors.processor_factory import ProcessorFactory; f=ProcessorFactory(); print('Factory created')\""
    - expect: "Factory created"
```

### Task 16: Create Main Service
```yaml
create_main_service:
  action: CREATE
  file: src/features/github/services/github_service.py
  changes: |
    Create main GitHubService orchestrator:
    - Implement dependency injection
    - Coordinate between all components
    - Maintain backward compatibility
    - Add comprehensive error handling
    - Include progress reporting
  validation:
    - command: "python -c \"from src.features.github.services.github_service import GitHubService; print('Service created')\""
    - expect: "Service created"
```

### Task 17: Update Module Initialization
```yaml
update_module_initialization:
  action: MODIFY
  file: src/features/github/__init__.py
  changes: |
    Export main public interfaces:
    - GitHubService as main entry point
    - Key interfaces for extensibility
    - Main data models
    - Configuration classes
    Maintain backward compatibility by re-exporting old classes
  validation:
    - command: "python -c \"from src.features.github import GitHubService; print('Module init updated')\""
    - expect: "Module init updated"
```

### Task 18: Update Import References
```yaml
update_import_references:
  action: MODIFY
  files:
    - src/tools/github_tools.py
    - src/services/unified_indexing_service.py
    - src/utils/__init__.py
  changes: |
    Update import statements to use new modular structure:
    - Change from single-file imports to module imports
    - Use GitHubService as main entry point
    - Update instantiation patterns
    - Maintain API compatibility
  validation:
    - command: "python -c \"import src.tools.github_tools; print('Imports updated')\""
    - expect: "Imports updated"
```

### Task 19: Update Tests
```yaml
update_tests:
  action: MODIFY
  files:
    - tests/features/test_github_processor.py
    - tests/github/test_github_processor.py
    - tests/test_github_processor.py
  changes: |
    Update test imports and instantiation:
    - Update import paths for new structure
    - Modify test instantiation to use new modules
    - Add new tests for individual components
    - Ensure all existing tests pass
  validation:
    - command: "uv run pytest tests/features/test_github_processor.py -v"
    - expect: "All tests pass"
```

### Task 20: Run Comprehensive Tests
```yaml
run_comprehensive_tests:
  action: MODIFY
  files: []
  changes: |
    Run full test suite to verify refactoring:
    - Run all GitHub processor related tests
    - Run integration tests
    - Verify no regressions in functionality
  validation:
    - command: "uv run pytest tests/ -k github_processor -v"
    - expect: "All tests pass without errors"
```

### Task 21: Cleanup Original File
```yaml
cleanup_original_file:
  action: DELETE
  file: src/features/github_processor.py
  changes: |
    Remove original monolithic file after verifying:
    - All functionality migrated
    - All tests passing
    - All imports updated
  validation:
    - command: "uv run pytest"
    - expect: "All tests continue to pass"
```

## Risk Assessment and Mitigation

### High Risk
- **Breaking Changes**: Import path changes could break existing code
  - **Mitigation**: Maintain backward compatibility through __init__.py re-exports
  - **Rollback**: Keep original file until verification complete

### Medium Risk  
- **Complex Dependencies**: Circular imports between new modules
  - **Mitigation**: Use dependency injection and careful module design
  - **Rollback**: Restructure imports if circular dependencies detected

### Low Risk
- **Performance Impact**: Multiple small files vs single large file
  - **Mitigation**: Python's import caching minimizes impact
  - **Rollback**: Performance regression unlikely

## Rollback Strategy

### Immediate Rollback (During Development)
1. Keep original file as `.py.backup` during refactoring
2. Revert changes if any task fails validation
3. Use Git branches for each phase

### Complete Rollback (After Deployment)
1. Restore original `github_processor.py`
2. Revert import changes in dependent files
3. Remove new modular structure
4. Re-run tests to ensure original functionality

## Quality Checklist

- [ ] **Current state fully documented**: ✅ Analyzed 1394-line monolithic file
- [ ] **Desired state clearly defined**: ✅ Modular architecture with clear separation
- [ ] **All objectives measurable**: ✅ Each task has specific validation criteria
- [ ] **Tasks ordered by dependency**: ✅ 6-phase approach with clear dependencies  
- [ ] **Each task has validation that AI can run**: ✅ Python import tests and pytest commands
- [ ] **Risks identified with mitigations**: ✅ High/Medium/Low risks with specific mitigations
- [ ] **Rollback strategy included**: ✅ Immediate and complete rollback procedures
- [ ] **Integration points noted**: ✅ All dependent files and test files identified

## Success Metrics

### Code Quality
- [ ] No file exceeds 500 lines (CLAUDE.md compliance)
- [ ] All classes follow Single Responsibility Principle
- [ ] Clear dependency injection throughout
- [ ] Comprehensive error handling

### Functionality
- [ ] All existing tests pass
- [ ] No breaking changes to public API
- [ ] All integration points working
- [ ] Performance maintained or improved

### Maintainability  
- [ ] Easy to add new file processors
- [ ] Clear module boundaries
- [ ] Comprehensive documentation
- [ ] Type hints throughout

### Testing
- [ ] Individual components testable in isolation
- [ ] Test coverage maintained or improved
- [ ] Integration tests pass
- [ ] Performance tests show no regression

---

**PRP Status**: Ready for Implementation  
**Estimated Effort**: 8-12 hours  
**Implementation Order**: Sequential phases with validation at each step  
**Risk Level**: Medium (due to complex refactoring scope)  