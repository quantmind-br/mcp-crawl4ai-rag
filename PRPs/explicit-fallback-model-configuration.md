# PRP: Explicit Fallback Model Configuration

**Status**: Draft  
**Created**: 2025-01-02  
**Type**: Specification-driven PRP  
**Complexity**: Medium  
**Domain**: Configuration Management  

## Overview

This PRP addresses the need to make fallback model configuration explicit in environment files (.env, .env.example) for better transparency and maintainability of the system's model selection logic.

## Current State Assessment

### Files Affected
- `.env` - Production environment configuration
- `.env.example` - Template environment configuration  
- `src/utils.py` - Contains implicit fallback logic in multiple functions
- `knowledge_graphs/test_script.py` - Uses fallback pattern

### Current Behavior
The system currently uses implicit fallback logic scattered throughout the codebase:

```python
# Current implicit fallback patterns found:
# Pattern 1: CHAT_MODEL → MODEL_CHOICE → hardcoded default
model_choice = os.getenv("CHAT_MODEL") or os.getenv("MODEL_CHOICE") or "gpt-4o-mini"

# Pattern 2: CHAT_MODEL with legacy fallback
model_choice = os.getenv("CHAT_MODEL", os.getenv("MODEL_CHOICE"))

# Pattern 3: API key fallbacks  
api_key = os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
api_key = os.getenv("EMBEDDINGS_API_KEY") or os.getenv("OPENAI_API_KEY")
```

### Issues Identified
1. **Hidden fallback logic**: Fallback models are hardcoded in functions rather than configured
2. **Inconsistent defaults**: Different functions use different hardcoded defaults
3. **Poor visibility**: Users cannot see what fallback models will be used without reading code
4. **Configuration drift**: No central place to configure system-wide fallback behavior
5. **Documentation gap**: Environment files don't document fallback mechanism

## Desired State Research

### Best Practices for Fallback Configuration
- **Explicit over implicit**: Make all fallback values visible in configuration
- **Single source of truth**: Centralize fallback configuration
- **Documentation**: Clear comments explaining fallback hierarchy
- **Validation**: Validate configuration chain at startup

### Implementation Examples
Other projects handle this by:
- Using explicit `*_FALLBACK` environment variables
- Documenting fallback chains in configuration files
- Providing validation tools for configuration

## Desired State Specification

### Files Structure
```yaml
.env.example:
  behavior: "Complete template with all fallback variables documented"
  benefits: ["Clear fallback hierarchy", "Better onboarding", "Reduced configuration errors"]

.env:
  behavior: "Production config with explicit fallback values set"
  benefits: ["Transparent model selection", "Predictable behavior", "Easy troubleshooting"]

src/utils.py:
  behavior: "Functions use environment variables for all fallbacks"
  benefits: ["Configurable behavior", "Consistent defaults", "Easier testing"]
```

### Configuration Hierarchy
```
CHAT_MODEL → CHAT_FALLBACK_MODEL → OPENAI_DEFAULT_MODEL → "gpt-4o-mini"
EMBEDDINGS_MODEL → EMBEDDINGS_FALLBACK_MODEL → OPENAI_DEFAULT_MODEL → "text-embedding-3-small"
```

## Implementation Strategy

### Phase 1: Environment Configuration (Priority: High)
- Add explicit fallback variables to .env.example
- Update .env with current values
- Document fallback hierarchy

### Phase 2: Code Updates (Priority: High)  
- Update utils.py functions to use explicit fallback variables
- Maintain backward compatibility during transition

### Phase 3: Validation (Priority: Medium)
- Add configuration validation
- Update tests to verify fallback behavior

## Hierarchical Objectives

### High-Level Goal
**Make fallback model configuration explicit and transparent across the entire system**

### Mid-Level Milestones
1. **Environment Configuration Updated**: Both .env files contain explicit fallback variables
2. **Code Integration Complete**: All functions use environment-based fallbacks
3. **Documentation Complete**: Clear fallback hierarchy documented
4. **Backward Compatibility Maintained**: Existing configurations continue to work

### Low-Level Tasks

#### Environment Configuration Tasks

```yaml
add_fallback_env_vars:
  action: ADD
  file: .env.example
  changes: |
    - Add CHAT_FALLBACK_MODEL variable with documentation
    - Add EMBEDDINGS_FALLBACK_MODEL variable with documentation  
    - Add OPENAI_DEFAULT_MODEL variable with documentation
    - Document complete fallback hierarchy in comments
  validation:
    - command: "grep -E '(FALLBACK_MODEL|DEFAULT_MODEL)' .env.example"
    - expect: "All three new variables present with documentation"

update_production_env:
  action: MODIFY  
  file: .env
  changes: |
    - Set CHAT_FALLBACK_MODEL=gpt-4o-mini (current hardcoded default)
    - Set EMBEDDINGS_FALLBACK_MODEL=text-embedding-3-small  
    - Set OPENAI_DEFAULT_MODEL=gpt-4o-mini
    - Add brief comments explaining fallback purpose
  validation:
    - command: "grep -E 'FALLBACK_MODEL|DEFAULT_MODEL' .env"
    - expect: "All fallback variables set with appropriate values"
```

#### Code Integration Tasks

```yaml
update_contextual_embedding_function:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 264: Update fallback chain to use CHAT_FALLBACK_MODEL
    - Replace line 264: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("MODEL_CHOICE") or os.getenv("OPENAI_DEFAULT_MODEL") or "gpt-4o-mini"
    - Add deprecation warning for MODEL_CHOICE usage
  validation:
    - command: "python -c \"from src.utils import generate_contextual_embedding; print('Function loads successfully')\""
    - expect: "No import errors, function accessible"

update_code_summary_function:
  action: MODIFY
  file: src/utils.py  
  changes: |
    - Replace line 604: Update fallback chain to use explicit variables
    - Replace line 604: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("MODEL_CHOICE") or os.getenv("OPENAI_DEFAULT_MODEL")
  validation:
    - command: "python -c \"from src.utils import generate_code_example_summary; print('Function loads successfully')\""
    - expect: "No import errors, function accessible"

update_source_summary_function:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 750: Update fallback chain to use explicit variables  
    - Replace line 750: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("MODEL_CHOICE") or os.getenv("OPENAI_DEFAULT_MODEL")
  validation:
    - command: "python -c \"from src.utils import extract_source_summary; print('Function loads successfully')\""
    - expect: "No import errors, function accessible"

update_test_script:
  action: MODIFY
  file: knowledge_graphs/test_script.py
  changes: |
    - Replace line 29: Update fallback chain to use explicit variables
    - Replace line 29: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or os.getenv("MODEL_CHOICE") or os.getenv("OPENAI_DEFAULT_MODEL") or "gpt-4o-mini"
  validation:
    - command: "python -m py_compile knowledge_graphs/test_script.py"
    - expect: "No syntax errors in compilation"
```

#### Documentation Tasks

```yaml
add_fallback_documentation:
  action: MODIFY
  file: .env.example
  changes: |
    - Add comprehensive section documenting fallback model hierarchy
    - Explain when each fallback level is used
    - Provide examples of different configuration scenarios
    - Add troubleshooting notes for model selection
  validation:
    - command: "grep -A 10 -B 2 'FALLBACK' .env.example"
    - expect: "Clear documentation of fallback system present"
```

## Risk Assessment & Mitigations

### Identified Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| **Breaking existing configurations** | Medium | High | Maintain backward compatibility, gradual migration |
| **Inconsistent fallback behavior** | Low | Medium | Comprehensive testing of all fallback scenarios |
| **Configuration confusion** | Low | Low | Clear documentation and examples |

### Rollback Strategy
1. Revert environment file changes
2. Restore original hardcoded fallback logic  
3. Remove new environment variables
4. Validate system returns to original behavior

## Validation & Testing

### Configuration Validation
```bash
# Test fallback hierarchy works correctly
export CHAT_MODEL=""
export CHAT_FALLBACK_MODEL="test-fallback"
python -c "from src.utils import generate_contextual_embedding; print('Testing fallback')"

# Test backward compatibility  
export CHAT_MODEL=""
export MODEL_CHOICE="test-legacy"
python -c "from src.utils import generate_contextual_embedding; print('Testing legacy')"
```

### Integration Testing
- Verify all functions use new fallback variables
- Test with various environment configurations
- Confirm no hardcoded defaults remain in critical paths

## Success Criteria

- [ ] All fallback models explicitly configured in environment files
- [ ] Complete fallback hierarchy documented in .env.example  
- [ ] All utils.py functions use environment-based fallbacks
- [ ] Backward compatibility maintained for existing MODEL_CHOICE usage
- [ ] Configuration validation passes with new variables
- [ ] No hardcoded model defaults remain in fallback chains

## Implementation Timeline

1. **Phase 1** (Day 1): Update environment files with fallback configuration
2. **Phase 2** (Day 1-2): Update code to use explicit fallback variables  
3. **Phase 3** (Day 2): Add comprehensive documentation and validation
4. **Testing** (Day 3): Comprehensive testing of all fallback scenarios
5. **Deployment** (Day 3): Deploy with validation and monitoring

## Dependencies & Prerequisites

- Write access to environment files
- Ability to modify source code
- Python environment for testing configuration changes
- Understanding of current model selection logic

## Integration Points

- **Environment Configuration**: Must integrate with existing .env structure
- **Model Selection Functions**: All functions in utils.py that select models
- **Testing Framework**: Existing tests should continue to pass
- **Documentation**: Align with existing documentation standards

---

**Implementation Notes:**
- Maintain strict backward compatibility during transition
- Use deprecation warnings for old patterns
- Test thoroughly with different environment configurations
- Document migration path for existing deployments