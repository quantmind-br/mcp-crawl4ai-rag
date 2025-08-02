# PRP: Remove OPENAI_DEFAULT_MODEL Redundancy

**Specification**: Remove the OPENAI_DEFAULT_MODEL configuration as it is redundant with CHAT_FALLBACK_MODEL and simplify configuration while maintaining same functionality

**Created**: 2025-01-02  
**Status**: Draft  
**Priority**: Medium  
**Type**: Configuration Simplification  

## Current State Assessment

### Files Affected
```yaml
configuration_files:
  - .env (line 25)
  - .env.example (line 27)

implementation_files:
  - src/utils.py (3 functions)
  - knowledge_graphs/test_script.py (1 function)

documentation_files:
  - PRPs/remove-legacy-model-configuration.md
  - PRPs/explicit-fallback-model-configuration.md
```

### Current Behavior
```yaml
fallback_hierarchy:
  level_1: CHAT_MODEL
  level_2: CHAT_FALLBACK_MODEL 
  level_3: OPENAI_DEFAULT_MODEL  # ← REDUNDANT
  level_4: "gpt-4o-mini" (hardcoded)

current_values:
  CHAT_FALLBACK_MODEL: "gpt-4.1-nano"
  OPENAI_DEFAULT_MODEL: "gpt-4.1-nano"  # ← IDENTICAL VALUE

functions_using_fallback:
  - generate_contextual_embedding()
  - generate_code_example_summary() 
  - extract_source_summary()
  - get_model() (knowledge graphs)
```

### Identified Issues
1. **Configuration Redundancy**: Two variables with identical values
2. **Maintenance Overhead**: Need to update two places for same change
3. **Cognitive Load**: Users confused about difference between variables
4. **Naming Inconsistency**: `OPENAI_DEFAULT_MODEL` suggests OpenAI-specific but used generally

## Desired State Research

### Target Architecture
```yaml
simplified_hierarchy:
  level_1: CHAT_MODEL
  level_2: CHAT_FALLBACK_MODEL  # Single fallback layer
  level_3: "gpt-4o-mini" (hardcoded)

configuration_reduction:
  removed_variables: ["OPENAI_DEFAULT_MODEL"]
  maintained_functionality: "100%"
  simplified_logic: "33% fewer fallback levels"
```

### Implementation Strategy
```yaml
approach: "Remove and Replace"
method: "Direct substitution in fallback chain"
risk_level: "Low" 
backward_compatibility: "Breaking change (acceptable for cleanup)"
```

## Hierarchical Objectives

### High-Level Goal
**Simplify model fallback configuration by removing redundant OPENAI_DEFAULT_MODEL variable while maintaining identical functionality**

### Mid-Level Milestones
1. **Environment Cleanup**: Remove OPENAI_DEFAULT_MODEL from .env files
2. **Code Modernization**: Update fallback logic to skip OPENAI_DEFAULT_MODEL
3. **Validation**: Ensure all functions maintain same behavior
4. **Documentation**: Update related documentation

### Low-Level Tasks

#### Task 1: Environment Configuration Cleanup
```yaml
task_name: "remove_openai_default_from_env"
action: MODIFY
files:
  - .env
  - .env.example
changes: |
  - Remove line: OPENAI_DEFAULT_MODEL=gpt-4.1-nano
  - Keep: CHAT_FALLBACK_MODEL=gpt-4.1-nano
  - Update comments if referencing removed variable
validation:
  command: "grep -n OPENAI_DEFAULT_MODEL .env .env.example"
  expect: "No matches found"
priority: 1
dependencies: []
```

#### Task 2: Update Utils Fallback Logic
```yaml
task_name: "simplify_utils_fallback"
action: MODIFY
file: "src/utils.py"
changes: |
  - Function: generate_contextual_embedding()
    FROM: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL") or "gpt-4o-mini"
    TO: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
  
  - Function: generate_code_example_summary()
    FROM: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL")
    TO: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL")
  
  - Function: extract_source_summary()
    FROM: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("OPENAI_DEFAULT_MODEL")
    TO: os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL")
validation:
  command: "python -c \"import sys; sys.path.insert(0, 'src'); from utils import generate_contextual_embedding; print('Fallback logic working')\""
  expect: "Fallback logic working"
priority: 2
dependencies: ["remove_openai_default_from_env"]
```

#### Task 3: Update Knowledge Graph Script
```yaml
task_name: "simplify_knowledge_graph_fallback"
action: MODIFY
file: "knowledge_graphs/test_script.py"
changes: |
  - Function: get_model()
    FROM: os.getenv('CHAT_MODEL') or os.getenv('CHAT_FALLBACK_MODEL') or os.getenv('OPENAI_DEFAULT_MODEL') or 'gpt-4o-mini'
    TO: os.getenv('CHAT_MODEL') or os.getenv('CHAT_FALLBACK_MODEL') or 'gpt-4o-mini'
validation:
  command: "python -c \"import sys; sys.path.insert(0, 'knowledge_graphs'); from test_script import get_model; print('Knowledge graph fallback working')\""
  expect: "Knowledge graph fallback working"
priority: 3
dependencies: ["remove_openai_default_from_env"]
```

#### Task 4: Comprehensive Validation
```yaml
task_name: "validate_no_references"
action: DELETE
changes: |
  - Ensure no remaining references to OPENAI_DEFAULT_MODEL exist
  - Verify fallback behavior is identical
  - Test configuration loading
validation:
  command: "grep -r 'OPENAI_DEFAULT_MODEL' . --exclude-dir=.git --exclude='*.backup' --exclude='PRPs/*.md'"
  expect: "No matches found (except this PRP and backup files)"
priority: 4
dependencies: ["simplify_utils_fallback", "simplify_knowledge_graph_fallback"]
```

## Implementation Order

```yaml
execution_sequence:
  1. remove_openai_default_from_env
  2. simplify_utils_fallback 
  3. simplify_knowledge_graph_fallback
  4. validate_no_references

parallel_execution:
  - Tasks 2 and 3 can run in parallel after Task 1
  - Task 4 must run after all others complete
```

## Risk Assessment

### Identified Risks
```yaml
configuration_risk:
  probability: "Low"
  impact: "Low" 
  description: "Users may have scripts depending on OPENAI_DEFAULT_MODEL"
  mitigation: "Breaking change is acceptable for cleanup, document in changelog"

functionality_risk:
  probability: "Very Low"
  impact: "Medium"
  description: "Fallback behavior could change unexpectedly"
  mitigation: "Comprehensive testing of all fallback scenarios"

testing_risk:
  probability: "Low"
  impact: "Low"
  description: "Tests may reference removed variable"
  mitigation: "Validate with grep search and run test suite"
```

### Go/No-Go Criteria
```yaml
proceed_if:
  - All validation commands pass
  - No external dependencies on OPENAI_DEFAULT_MODEL found
  - Test suite continues to pass
  
abort_if:
  - Critical external integrations depend on variable
  - Fallback behavior changes unexpectedly
  - Test failures cannot be resolved
```

## Rollback Strategy

```yaml
rollback_plan:
  1. Restore OPENAI_DEFAULT_MODEL to .env files
  2. Restore fallback logic in source files  
  3. Run validation tests
  4. Verify original behavior restored

rollback_files:
  - .env.backup (contains original configuration)
  - Git history (track all changes for easy revert)
  
rollback_time: "< 5 minutes"
rollback_complexity: "Simple (direct revert)"
```

## Benefits Analysis

### Configuration Benefits
```yaml
simplification:
  - 1 fewer environment variable to configure
  - 1 fewer fallback level to understand
  - Clearer configuration hierarchy

maintenance:
  - Single point of change for fallback model
  - Reduced cognitive load for configuration
  - Elimination of value synchronization issues
```

### Code Benefits  
```yaml
readability:
  - Shorter fallback chains
  - Clearer intent in variable naming
  - Reduced complexity in model selection logic

performance:
  - Minimal: 1 fewer os.getenv() call per fallback
  - Negligible impact but technically more efficient
```

## Validation Criteria

### Success Metrics
```yaml
functionality:
  - All model selection functions work identically
  - Fallback behavior unchanged from user perspective
  - No runtime errors in any usage scenario

cleanliness:
  - Zero references to OPENAI_DEFAULT_MODEL in active code
  - Configuration files contain only necessary variables
  - Documentation reflects simplified hierarchy

testing:
  - All existing tests continue to pass
  - Fallback logic tests validate new hierarchy
  - Configuration validation works correctly
```

## Quality Checklist

- [x] Current state fully documented
- [x] Desired state clearly defined  
- [x] All objectives measurable
- [x] Tasks ordered by dependency
- [x] Each task has validation command
- [x] Risks identified with mitigations
- [x] Rollback strategy included
- [x] Integration points noted

## Execution Notes

This is a **low-risk, high-value cleanup** that removes configuration redundancy without changing functionality. The transformation is straightforward with clear validation at each step.

**Estimated Execution Time**: 15 minutes  
**Complexity**: Low  
**Breaking Change**: Yes (acceptable for cleanup)  
**User Impact**: Positive (simpler configuration)