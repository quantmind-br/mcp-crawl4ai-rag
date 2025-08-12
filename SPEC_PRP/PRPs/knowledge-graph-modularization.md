# PRP: Knowledge Graph Modularization

## Executive Summary

**Goal**: Refactor the `knowledge_graphs` directory into the main `src` directory to improve architectural consistency, modularity, and maintainability.

**Type**: Structural Refactoring  
**Priority**: High  
**Estimated Effort**: Medium (3-5 days)

## Current State Analysis

```yaml
current_state:
  structure: |
    knowledge_graphs/                    # Root-level directory (architectural inconsistency)
    ├── __init__.py
    ├── ai_hallucination_detector.py    # Main entry point for hallucination detection
    ├── ai_script_analyzer.py           # Core analysis logic + data models
    ├── build_grammars.py              # Utility script
    ├── hallucination_reporter.py      # Reporting functionality
    ├── knowledge_graph_validator.py   # Validation logic
    ├── language_parser.py             # ABC interface + data models
    ├── parser_factory.py              # Parser creation logic
    ├── parse_repo_into_neo4j.py       # Repository parsing service
    ├── query_knowledge_graph.py       # Utility script
    ├── simple_fallback_parser.py      # Parser implementation
    ├── tree_sitter_parser.py          # Parser implementation
    └── query_patterns/                # Language-specific queries
        ├── __init__.py
        ├── c_queries.py
        ├── go_queries.py
        ├── java_queries.py
        ├── javascript_queries.py
        ├── python_queries.py
        └── typescript_queries.py
  
  dependencies: |
    External dependencies:
    - src/tools/kg_tools.py -> knowledge_graphs.*
    - src/core/app.py -> knowledge_graphs.*
    - src/utils/grammar_initialization.py -> knowledge_graphs.*
    - debug_*.py scripts -> knowledge_graphs.*
  
  issues:
    - architectural_inconsistency: "Breaks src/ directory convention"
    - tight_coupling: "Direct cross-boundary dependencies from src/ to root-level package"
    - reduced_cohesion: "Mixed responsibilities without clear separation"
    - testing_difficulty: "Monolithic structure impedes unit testing"

desired_state:
  structure: |
    src/k_graph/                        # Follows src/ directory convention
    ├── __init__.py
    ├── analysis/                       # High-level analysis components
    │   ├── __init__.py
    │   ├── hallucination_detector.py   # Main detection entry point
    │   ├── reporter.py                 # Reporting functionality
    │   ├── script_analyzer.py          # Analysis logic (implementation only)
    │   └── validator.py                # Validation logic
    ├── core/                           # Shared abstractions
    │   ├── __init__.py
    │   ├── interfaces.py               # Abstract base classes (LanguageParser)
    │   └── models.py                   # Data models (ParseResult, ImportInfo, etc.)
    ├── parsing/                        # Tree-sitter parsing engine
    │   ├── __init__.py
    │   ├── grammars/                   # Grammar files (moved)
    │   ├── parser_factory.py          # Parser creation logic
    │   ├── query_patterns/             # Language-specific queries (moved)
    │   ├── simple_fallback_parser.py   # Parser implementation
    │   └── tree_sitter_parser.py       # Parser implementation
    └── services/                       # External system services
        ├── __init__.py
        └── repository_parser.py        # Neo4j integration service
    
    scripts/                            # Project utility scripts
    ├── build_grammars.py              # Moved from knowledge_graphs/
    └── query_knowledge_graph.py       # Moved from knowledge_graphs/
  
  benefits:
    - architectural_consistency: "All application code under src/"
    - improved_modularity: "Clear separation of concerns"
    - better_testability: "Smaller, focused components"
    - enhanced_maintainability: "Logical organization and dependencies"
```

## Hierarchical Objectives

### 1. High-Level: Structural Reorganization
**Goal**: Move knowledge_graphs functionality into src/k_graph with proper modular structure
**Success Criteria**: All functionality preserved, architectural consistency achieved

### 2. Mid-Level: Component Separation
**Goal**: Decompose monolithic files into focused, single-responsibility components
**Success Criteria**: Clear interfaces, separated concerns, improved testability

### 3. Low-Level: Import Path Updates
**Goal**: Update all import statements and path logic to reflect new structure
**Success Criteria**: No broken imports, all tests pass, application runs successfully

## Detailed Implementation Plan

### Phase 1: Scaffolding and Core Setup

#### Task 1.1: Create Directory Structure
```yaml
action: CREATE
priority: P0 (Prerequisite)
files:
  - src/k_graph/__init__.py
  - src/k_graph/analysis/__init__.py
  - src/k_graph/core/__init__.py
  - src/k_graph/parsing/__init__.py
  - src/k_graph/parsing/query_patterns/__init__.py
  - src/k_graph/services/__init__.py
changes: |
  - Create complete src/k_graph directory tree
  - Add appropriate __init__.py files with docstrings
validation:
  command: "ls -la src/k_graph/"
  expect: "Directory structure created successfully"
```

#### Task 1.2: Create Core Abstractions
```yaml
action: CREATE
priority: P0 (Prerequisite)
files:
  - src/k_graph/core/models.py
  - src/k_graph/core/interfaces.py
changes: |
  - Extract data models from knowledge_graphs/language_parser.py
  - Extract LanguageParser ABC to interfaces.py
  - Move ParseResult, ImportInfo, ClassInfo, etc. to models.py
validation:
  command: "python -c 'from src.k_graph.core.models import ParseResult'"
  expect: "Import successful"
```

### Phase 2: File Migration and Decomposition

#### Task 2.1: Migrate Analysis Components
```yaml
action: MOVE + MODIFY
priority: P1
mappings:
  - knowledge_graphs/ai_script_analyzer.py -> src/k_graph/analysis/script_analyzer.py
  - knowledge_graphs/knowledge_graph_validator.py -> src/k_graph/analysis/validator.py
  - knowledge_graphs/hallucination_reporter.py -> src/k_graph/analysis/reporter.py
  - knowledge_graphs/ai_hallucination_detector.py -> src/k_graph/analysis/hallucination_detector.py
changes: |
  - Move files to new locations
  - Extract data models to core/models.py (remove duplicates)
  - Update internal imports to use core.models
validation:
  command: "python -c 'from src.k_graph.analysis.script_analyzer import AIScriptAnalyzer'"
  expect: "Import successful without data model definitions"
```

#### Task 2.2: Migrate Parsing Engine
```yaml
action: MOVE + MODIFY
priority: P1
mappings:
  - knowledge_graphs/parser_factory.py -> src/k_graph/parsing/parser_factory.py
  - knowledge_graphs/tree_sitter_parser.py -> src/k_graph/parsing/tree_sitter_parser.py
  - knowledge_graphs/simple_fallback_parser.py -> src/k_graph/parsing/simple_fallback_parser.py
  - knowledge_graphs/query_patterns/ -> src/k_graph/parsing/query_patterns/
  - knowledge_graphs/grammars/ -> src/k_graph/parsing/grammars/ (if exists)
changes: |
  - Move parsing implementation files
  - Extract LanguageParser ABC to core/interfaces.py
  - Update imports to use core.interfaces, core.models
validation:
  command: "python -c 'from src.k_graph.parsing.parser_factory import ParserFactory'"
  expect: "Import successful without ABC definition"
```

#### Task 2.3: Migrate Services
```yaml
action: MOVE + MODIFY
priority: P1
mappings:
  - knowledge_graphs/parse_repo_into_neo4j.py -> src/k_graph/services/repository_parser.py
changes: |
  - Move Neo4j integration service
  - Update imports to use src.k_graph.* paths
validation:
  command: "python -c 'from src.k_graph.services.repository_parser import parse_repository_into_neo4j'"
  expect: "Import successful"
```

#### Task 2.4: Migrate Utility Scripts
```yaml
action: MOVE + MODIFY
priority: P2
mappings:
  - knowledge_graphs/build_grammars.py -> scripts/build_grammars.py
  - knowledge_graphs/query_knowledge_graph.py -> scripts/query_knowledge_graph.py
changes: |
  - Move utility scripts to scripts/ directory
  - Update path logic to be project-root relative
  - Fix imports to use src.k_graph.* paths
validation:
  command: "python scripts/query_knowledge_graph.py --help"
  expect: "Script runs without import errors"
```

### Phase 3: Import Path Refactoring

#### Task 3.1: Update Internal k_graph Imports
```yaml
action: MODIFY
priority: P0 (Critical)
files: [all files in src/k_graph/]
changes: |
  - Replace "from knowledge_graphs" with "from src.k_graph"
  - Use relative imports where appropriate
  - Import from core.models and core.interfaces
validation:
  command: "grep -r 'knowledge_graphs' src/k_graph/"
  expect: "No matches found"
```

#### Task 3.2: Update External Dependencies
```yaml
action: MODIFY
priority: P0 (Critical)
files:
  - src/tools/kg_tools.py
  - src/core/app.py
  - src/utils/grammar_initialization.py
changes: |
  - Replace knowledge_graphs imports with src.k_graph imports
  - Update function calls to use new module structure
validation:
  command: "python -c 'import src.tools.kg_tools; import src.core.app'"
  expect: "Imports successful"
```

#### Task 3.3: Update Root-Level Scripts
```yaml
action: MODIFY
priority: P1
files: [debug_*.py files in project root]
changes: |
  - Update knowledge_graphs imports to src.k_graph
  - Verify script functionality still works
validation:
  command: "python -m py_compile debug_*.py"
  expect: "All scripts compile successfully"
```

#### Task 3.4: Fix Moved Script Path Logic
```yaml
action: MODIFY
priority: P1
files:
  - scripts/build_grammars.py
  - scripts/query_knowledge_graph.py
changes: |
  - Update grammars directory path to src/k_graph/parsing/grammars/
  - Use project root-relative paths instead of __file__-relative
  - Update all imports to src.k_graph.*
validation:
  command: "cd scripts && python build_grammars.py --dry-run"
  expect: "Correct grammars path resolved"
```

### Phase 4: Verification and Cleanup

#### Task 4.1: Run Code Quality Checks
```yaml
action: VALIDATE
priority: P0 (Critical)
commands:
  - "uv run ruff check . --fix"
  - "uv run ruff format ."
validation:
  expect: "No linting errors remain"
```

#### Task 4.2: Execute Test Suite
```yaml
action: VALIDATE
priority: P0 (Critical)
commands:
  - "uv run pytest tests/ -v"
validation:
  expect: "All tests pass"
```

#### Task 4.3: Manual Integration Test
```yaml
action: VALIDATE
priority: P0 (Critical)
commands:
  - "uv run -m src"  # Start server
  - Test kg_tools.py functions via MCP
validation:
  expect: "Server starts, tools execute successfully"
```

#### Task 4.4: Remove Old Directory
```yaml
action: DELETE
priority: P1 (Final step)
files: [knowledge_graphs/]
changes: |
  - Delete entire knowledge_graphs directory
  - Verify no remaining references
validation:
  command: "grep -r 'knowledge_graphs' . --exclude-dir=.git"
  expect: "Only references in this PRP document"
```

## Risk Assessment and Mitigation

### High-Risk Areas

#### 1. Import Path Updates (Critical Risk)
**Risk**: Missing import path updates could break application at runtime
**Mitigation**: 
- Systematic grep-based search for all "knowledge_graphs" references
- Phase-by-phase validation after each change
- Comprehensive test suite execution

#### 2. Path Logic in Scripts (Medium Risk)
**Risk**: Moved scripts may have broken relative path assumptions
**Mitigation**:
- Explicit task to refactor path resolution logic
- Use project-root relative paths instead of `__file__` relative
- Test script execution in validation

### Dependencies and Ordering
1. Phase 1 (Scaffolding) must complete before Phase 2 (Migration)
2. Phase 2 (Migration) must complete before Phase 3 (Refactoring)
3. Task 3.1 (Internal imports) must complete before Task 3.2 (External imports)
4. All implementation tasks must complete before Phase 4 (Cleanup)

## Success Metrics

### Quantitative Metrics
- [ ] 0 occurrences of "knowledge_graphs" in codebase (excluding documentation)
- [ ] 100% test pass rate maintained
- [ ] 0 linting errors after refactoring
- [ ] All MCP tools functional in integration test

### Qualitative Metrics
- [ ] Improved code organization with clear separation of concerns
- [ ] Consistent src/ directory architecture
- [ ] Enhanced maintainability through modular structure
- [ ] Better testability via focused components

## Rollback Strategy

### Immediate Rollback (During Implementation)
If any phase fails:
1. Stop current phase
2. Restore from git working state before phase started
3. Investigate and address root cause
4. Resume with corrected approach

### Complete Rollback (After Completion)
If issues discovered post-implementation:
1. `git revert` commits related to refactoring
2. Restore `knowledge_graphs/` directory from backup
3. Revert any test or configuration changes
4. Re-analyze approach before retry

## Validation Commands

```bash
# Structure validation
find src/k_graph -name "*.py" | wc -l  # Should match original file count

# Import validation  
python -c "import src.k_graph; print('k_graph package imports successfully')"

# Functionality validation
uv run pytest tests/knowledge_graphs/ -v  # Should pass if tests exist

# Integration validation
uv run -m src &  # Start server
# Test MCP tools via client

# Cleanup validation
find . -name "*.py" -exec grep -l "knowledge_graphs" {} \; | grep -v PRP
```

## Dependencies

### External Dependencies
- Existing test suite coverage for regression detection
- Git version control for rollback capability
- UV package manager for testing commands

### Internal Dependencies
- All phases must execute in specified order
- Core abstraction files must be created before component migration
- Import updates must be systematic and complete

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-12  
**Reviewed By**: Claude Code Assistant  
**Status**: Ready for Implementation