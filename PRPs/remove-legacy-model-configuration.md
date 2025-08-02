# PRP: Complete Removal of Legacy Model Configuration (MODEL_CHOICE & OPENAI_API_KEY)

**Status**: Draft  
**Created**: 2025-01-02  
**Type**: Specification-driven PRP  
**Complexity**: High  
**Domain**: Configuration Management & Technical Debt Removal  

## Overview

This PRP addresses the complete removal of legacy configuration variables `MODEL_CHOICE` and `OPENAI_API_KEY` from the entire codebase, transitioning to the modern `CHAT_MODEL`/`EMBEDDINGS_MODEL` + dedicated API key system.

## Current State Assessment

### Files Affected

**Primary Code Files:**
- `src/utils.py` - 13 references to MODEL_CHOICE, 7 references to OPENAI_API_KEY
- `knowledge_graphs/test_script.py` - 2 references each
- `.env` - Contains legacy variables
- `.env.example` - Documents legacy variables as deprecated

**Test Files:**
- `tests/test_flexible_api_config.py` - 14 references (backward compatibility tests)
- `tests/conftest.py` - Test environment setup
- `tests/test_mcp_basic.py` - 3 references  
- `tests/test_mcp_server.py` - 2 references
- `tests/test_integration_docker.py` - 2 references

### Current Behavior

**Legacy Fallback Patterns Found:**
```python
# Pattern 1: Dual fallback (src/utils.py lines 264, 604, 750)
model_choice = os.getenv("CHAT_MODEL") or os.getenv("MODEL_CHOICE") or "hardcoded-default"

# Pattern 2: Nested fallback (knowledge_graphs/test_script.py line 29)
model_choice = os.getenv('CHAT_MODEL', os.getenv('MODEL_CHOICE', 'gpt-4.1-mini'))

# Pattern 3: API key fallback (src/utils.py lines 41, 76)
api_key = os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
api_key = os.getenv("EMBEDDINGS_API_KEY") or os.getenv("OPENAI_API_KEY")
```

**Validation Logic Dependencies:**
```python
# validate_chat_config() - lines 102, 108, 113, 122-126
openai_api_key = os.getenv("OPENAI_API_KEY")
model_choice = os.getenv("MODEL_CHOICE")
# Includes deprecation warnings and error messages
```

### Issues Identified

1. **Technical Debt**: Legacy variables maintained across 8+ files
2. **Code Complexity**: Multiple fallback patterns creating confusion
3. **Testing Overhead**: Extensive backward compatibility test suite
4. **Configuration Bloat**: Duplicate environment variables in .env files
5. **Documentation Burden**: Comments explaining deprecated functionality
6. **Error Message Complexity**: References to both old and new variables

## Desired State Research

### Best Practices for Legacy Removal
- **Clean Break Approach**: Remove all references in single coordinated effort
- **Modern Configuration Only**: Use dedicated API keys for each service
- **Simplified Validation**: Single validation path without legacy checks
- **Updated Documentation**: Remove all references to deprecated variables

### Migration Strategy Examples
```python
# Current (complex fallback):
model_choice = os.getenv("CHAT_MODEL") or os.getenv("MODEL_CHOICE") or "gpt-4o-mini"

# Target (clean modern):
model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
```

## Desired State Specification

### Files Structure
```yaml
src/utils.py:
  behavior: "Clean modern configuration only - no legacy fallbacks"
  benefits: ["Simplified code", "Single responsibility", "Clear error messages"]

environment_files:
  behavior: "Modern variables only - CHAT_* and EMBEDDINGS_* only"
  benefits: ["Reduced confusion", "Cleaner configuration", "Single source of truth"]

test_files:
  behavior: "Test modern configuration paths only"
  benefits: ["Faster test execution", "Reduced test complexity", "Clear validation"]
```

### Modern Configuration Standard
```bash
# Only these variables will remain:
CHAT_MODEL=gemini/gemini-2.5-flash
CHAT_API_KEY=your-chat-api-key  
CHAT_API_BASE=https://api.quantmind.com.br/v1

EMBEDDINGS_MODEL=text-embedding-3-small
EMBEDDINGS_API_KEY=your-embeddings-api-key
EMBEDDINGS_API_BASE=https://api.openai.com/v1

# Fallback configuration (from previous PRP):
CHAT_FALLBACK_MODEL=gpt-4o-mini
EMBEDDINGS_FALLBACK_MODEL=text-embedding-3-small
OPENAI_DEFAULT_MODEL=gpt-4o-mini
```

## Risk Assessment & Mitigations

### Identified Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| **Breaking existing deployments** | High | Critical | Clear migration guide, validation script |
| **Test failures during transition** | Medium | High | Update tests before code, comprehensive validation |
| **Configuration errors in production** | Medium | High | Pre-deployment configuration validation |
| **Developer confusion** | Low | Medium | Clear documentation, migration checklist |

### Critical Dependencies
- **Docker integration tests**: Currently skip if `OPENAI_API_KEY` not present
- **Backward compatibility tests**: Entire test suite depends on legacy variables
- **Production environments**: May have only legacy variables configured

## Implementation Strategy

### Phase 1: Preparation & Validation (Priority: Critical)
- Create configuration validation script
- Document current production configurations  
- Prepare migration guide for deployments

### Phase 2: Environment & Documentation (Priority: High)
- Remove legacy variables from .env files
- Update all documentation and comments
- Remove deprecation warnings

### Phase 3: Code Modernization (Priority: High)
- Update all fallback logic to modern configuration
- Simplify validation functions
- Remove legacy error messages

### Phase 4: Test Suite Overhaul (Priority: Medium)
- Remove backward compatibility tests
- Update test configurations
- Add modern configuration validation tests

## Hierarchical Objectives

### High-Level Goal
**Complete removal of MODEL_CHOICE and OPENAI_API_KEY from entire codebase, transitioning to modern configuration system**

### Mid-Level Milestones
1. **Environment Cleaned**: All .env files contain only modern variables
2. **Code Modernized**: All fallback logic uses modern configuration pattern
3. **Tests Updated**: Test suite validates modern configuration only
4. **Documentation Updated**: No references to legacy variables remain

### Low-Level Tasks

#### Environment Configuration Tasks

```yaml
remove_legacy_from_env_example:
  action: DELETE
  file: .env.example
  changes: |
    - Remove MODEL_CHOICE variable and all references
    - Remove OPENAI_API_KEY variable and backward compatibility comments
    - Remove deprecation warnings about MODEL_CHOICE
    - Update comments to reference only modern variables
  validation:
    - command: "grep -E '(MODEL_CHOICE|OPENAI_API_KEY)' .env.example"
    - expect: "No matches found"

remove_legacy_from_production_env:
  action: DELETE
  file: .env
  changes: |
    - Remove MODEL_CHOICE=gpt-4o-mini line
    - Remove OPENAI_API_KEY (move value to CHAT_API_KEY if needed)
    - Ensure CHAT_MODEL and CHAT_API_KEY are properly configured
  validation:
    - command: "grep -E '(MODEL_CHOICE|OPENAI_API_KEY)' .env"
    - expect: "No matches found"
```

#### Code Modernization Tasks

```yaml
modernize_utils_generate_contextual_embedding:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 252: Remove reference to MODEL_CHOICE fallback in docstring
    - Replace line 263: Remove comment about MODEL_CHOICE fallback  
    - Replace line 264: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL") or "gpt-4o-mini"
    - Replace line 267: Update warning message to mention only CHAT_MODEL
  validation:
    - command: "grep -n 'MODEL_CHOICE' src/utils.py"
    - expect: "No matches in generate_contextual_embedding function"

modernize_utils_code_example_summary:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 593: Remove reference to MODEL_CHOICE fallback in docstring
    - Replace line 603: Remove comment about MODEL_CHOICE fallback
    - Replace line 604: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL")
  validation:
    - command: "grep -n 'MODEL_CHOICE' src/utils.py"
    - expect: "No matches in generate_code_example_summary function"

modernize_utils_source_summary:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 733: Remove reference to MODEL_CHOICE fallback in docstring
    - Replace line 749: Remove comment about MODEL_CHOICE fallback
    - Replace line 750: model_choice = os.getenv("CHAT_MODEL") or os.getenv("CHAT_FALLBACK_MODEL")
  validation:
    - command: "grep -n 'MODEL_CHOICE' src/utils.py"
    - expect: "No matches in extract_source_summary function"

modernize_chat_client_function:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 31: Update docstring to remove OPENAI_API_KEY fallback reference
    - Replace line 41: api_key = os.getenv("CHAT_API_KEY")
    - Replace line 46: Update error message to mention only CHAT_API_KEY
  validation:
    - command: "grep -n 'OPENAI_API_KEY' src/utils.py | grep -v 'validate_'"
    - expect: "No matches outside validation functions"

modernize_embeddings_client_function:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace line 66: Update docstring to remove OPENAI_API_KEY fallback reference
    - Replace line 76: api_key = os.getenv("EMBEDDINGS_API_KEY")
    - Replace line 81: Update error message to mention only EMBEDDINGS_API_KEY
  validation:
    - command: "grep -n 'OPENAI_API_KEY' src/utils.py | grep -v 'validate_'"
    - expect: "No matches outside validation functions"

remove_validation_legacy_support:
  action: MODIFY
  file: src/utils.py
  changes: |
    - Replace lines 102-109: Remove OPENAI_API_KEY checks and error messages
    - Replace line 113: Remove MODEL_CHOICE validation
    - Replace lines 122-126: Remove deprecation warning for MODEL_CHOICE
    - Replace line 129: Remove fallback source logging
    - Replace lines 149-156: Remove OPENAI_API_KEY validation for embeddings
    - Replace line 160: Remove fallback source logging
  validation:
    - command: "python -c \"from src.utils import validate_chat_config, validate_embeddings_config; print('Validation functions load successfully')\""
    - expect: "No import errors, functions accessible"

modernize_test_script:
  action: MODIFY
  file: knowledge_graphs/test_script.py
  changes: |
    - Replace line 28: Remove comment about MODEL_CHOICE fallback
    - Replace line 29: model_choice = os.getenv('CHAT_MODEL') or os.getenv('CHAT_FALLBACK_MODEL') or 'gpt-4o-mini'
    - Replace line 31: Remove comment about OPENAI_API_KEY fallback  
    - Replace line 32: api_key = os.getenv('CHAT_API_KEY') or 'no-api-key-provided'
  validation:
    - command: "python -m py_compile knowledge_graphs/test_script.py"
    - expect: "No syntax errors in compilation"

remove_legacy_openai_import:
  action: DELETE
  file: src/utils.py
  changes: |
    - Remove line 24: openai.api_key = os.getenv("OPENAI_API_KEY")
  validation:
    - command: "grep -n 'openai.api_key' src/utils.py"
    - expect: "No matches found"
```

#### Test Suite Modernization Tasks

```yaml
remove_backward_compatibility_tests:
  action: DELETE
  file: tests/test_flexible_api_config.py
  changes: |
    - Remove lines 76-87: test_backward_compatibility_chat_model function
    - Remove lines 88-97: test_backward_compatibility_embeddings function
    - Remove lines 99-121: test_model_choice_fallback_logic function
    - Remove lines 187-199: test_deprecation_warnings function
    - Remove all MODEL_CHOICE and OPENAI_API_KEY from setup/teardown
  validation:
    - command: "pytest tests/test_flexible_api_config.py -v"
    - expect: "All remaining tests pass without legacy variable dependencies"

update_conftest_environment:
  action: MODIFY
  file: tests/conftest.py
  changes: |
    - Replace line 20: Remove OPENAI_API_KEY from test environment
    - Replace line 21: Remove MODEL_CHOICE from test environment
    - Ensure CHAT_API_KEY and EMBEDDINGS_API_KEY are properly set
  validation:
    - command: "grep -E '(MODEL_CHOICE|OPENAI_API_KEY)' tests/conftest.py"
    - expect: "No matches found"

update_mcp_test_environments:
  action: MODIFY
  file: tests/test_mcp_basic.py
  changes: |
    - Replace line 18: Change OPENAI_API_KEY to CHAT_API_KEY
    - Replace line 19: Change MODEL_CHOICE to CHAT_MODEL
    - Update line 155-156: Remove legacy variables from environment list
  validation:
    - command: "pytest tests/test_mcp_basic.py -v"
    - expect: "All tests pass with modern configuration"

update_mcp_server_tests:
  action: MODIFY
  file: tests/test_mcp_server.py
  changes: |
    - Replace line 20: Change OPENAI_API_KEY to CHAT_API_KEY
    - Replace line 21: Change MODEL_CHOICE to CHAT_MODEL
  validation:
    - command: "pytest tests/test_mcp_server.py -v"
    - expect: "All tests pass with modern configuration"

update_integration_tests:
  action: MODIFY
  file: tests/test_integration_docker.py
  changes: |
    - Replace line 183: Change @pytest.mark.skipif condition to check CHAT_API_KEY
    - Replace line 287: Change @pytest.mark.skipif condition to check CHAT_API_KEY
  validation:
    - command: "python -c \"import tests.test_integration_docker; print('Module loads successfully')\""
    - expect: "No import errors"
```

## Migration Guide for Deployments

### Pre-Migration Checklist
```bash
# 1. Check current configuration
grep -E "(MODEL_CHOICE|OPENAI_API_KEY)" .env

# 2. Backup current configuration
cp .env .env.backup

# 3. Ensure modern variables are set
grep -E "(CHAT_MODEL|CHAT_API_KEY|EMBEDDINGS_MODEL|EMBEDDINGS_API_KEY)" .env
```

### Migration Script
```bash
#!/bin/bash
# migrate_to_modern_config.sh

echo "Migrating to modern configuration..."

# If OPENAI_API_KEY exists but CHAT_API_KEY doesn't, migrate it
if [ -n "$OPENAI_API_KEY" ] && [ -z "$CHAT_API_KEY" ]; then
    echo "CHAT_API_KEY=$OPENAI_API_KEY" >> .env
    echo "Migrated OPENAI_API_KEY to CHAT_API_KEY"
fi

# If OPENAI_API_KEY exists but EMBEDDINGS_API_KEY doesn't, migrate it  
if [ -n "$OPENAI_API_KEY" ] && [ -z "$EMBEDDINGS_API_KEY" ]; then
    echo "EMBEDDINGS_API_KEY=$OPENAI_API_KEY" >> .env
    echo "Migrated OPENAI_API_KEY to EMBEDDINGS_API_KEY"
fi

# If MODEL_CHOICE exists but CHAT_MODEL doesn't, warn user
if [ -n "$MODEL_CHOICE" ] && [ -z "$CHAT_MODEL" ]; then
    echo "WARNING: MODEL_CHOICE found but CHAT_MODEL not set."
    echo "Please set CHAT_MODEL manually based on your requirements."
    echo "MODEL_CHOICE value: $MODEL_CHOICE"
fi

echo "Migration preparation complete. Please review .env file."
```

## Rollback Strategy

### Emergency Rollback Process
1. **Restore .env files** from backup
2. **Revert code changes** using git
3. **Restore test configurations**
4. **Validate system functionality**

### Rollback Validation
```bash
# Verify legacy variables are recognized
python -c "import os; os.environ['MODEL_CHOICE']='test'; from src.utils import generate_contextual_embedding; print('Legacy support restored')"
```

## Validation & Testing

### Pre-Implementation Validation
```bash
# Ensure all modern variables are configured
python -c "
import os
required = ['CHAT_MODEL', 'CHAT_API_KEY', 'EMBEDDINGS_MODEL', 'EMBEDDINGS_API_KEY']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'Missing required variables: {missing}')
    exit(1)
print('All required modern variables configured')
"
```

### Post-Implementation Testing
```bash
# Test all functions work with modern configuration only
python -c "
from src.utils import get_chat_client, get_embeddings_client, generate_contextual_embedding
from knowledge_graphs.test_script import main
print('All imports successful')
"

# Run test suite
pytest tests/ -v --tb=short
```

## Success Criteria

- [ ] Zero references to MODEL_CHOICE in entire codebase
- [ ] Zero references to OPENAI_API_KEY outside migration documentation
- [ ] All tests pass with modern configuration only
- [ ] All functions work without legacy variable fallbacks
- [ ] Environment files contain only modern variables
- [ ] Documentation updated to reflect modern configuration
- [ ] Migration guide provided for existing deployments
- [ ] Rollback strategy validated and documented

## Implementation Timeline

1. **Phase 1** (Day 1): Create migration script and validate current deployments
2. **Phase 2** (Day 1): Update environment files and documentation
3. **Phase 3** (Day 2): Modernize all code functions and remove legacy logic
4. **Phase 4** (Day 2): Update entire test suite to modern configuration
5. **Validation** (Day 3): Comprehensive testing and validation
6. **Deployment** (Day 3): Deploy with migration guide and monitoring

## Dependencies & Prerequisites

- **Modern configuration fully functional**: CHAT_MODEL, CHAT_API_KEY, etc. must work perfectly
- **Fallback configuration implemented**: Previous PRP for explicit fallbacks should be completed first
- **Deployment coordination**: All production environments need migration planning
- **Test environment access**: Ability to run full test suite during migration

## Integration Points

- **Environment Configuration**: Must maintain system functionality during transition
- **OpenAI Client Creation**: All client creation must work with modern variables only
- **Model Selection Logic**: All model selection must use modern fallback hierarchy
- **Test Framework**: Tests must validate modern configuration paths only
- **Production Deployments**: Must not break existing live systems

---

**Implementation Notes:**
- **BREAKING CHANGE**: This removes backward compatibility entirely
- **Coordination Required**: All deployments must migrate simultaneously
- **Testing Critical**: Comprehensive testing required before deployment
- **Documentation Essential**: Clear migration path for all stakeholders