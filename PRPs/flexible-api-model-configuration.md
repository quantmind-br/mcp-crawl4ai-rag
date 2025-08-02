# PRP: Flexible API and Model Configuration

## Summary
Transform the current hardcoded OpenAI API configuration into a flexible system that allows users to configure any OpenAI-compatible API for both chat and embeddings models through environment variables.

## Current State Assessment

### Existing Implementation
```yaml
current_state:
  files:
    - .env.example: Contains MODEL_CHOICE and OPENAI_API_KEY
    - src/utils.py: Uses MODEL_CHOICE and hardcoded OpenAI client
    - Various test files: Set MODEL_CHOICE environment variable
  behavior: 
    - Chat model configured via MODEL_CHOICE environment variable
    - Embeddings always use OpenAI API with OPENAI_API_KEY
    - Base URL hardcoded to OpenAI's endpoints
    - No flexibility for alternative OpenAI-compatible APIs
  issues:
    - Limited to OpenAI API only
    - No way to configure custom base URLs
    - Embeddings model not configurable
    - Inconsistent naming convention (MODEL_CHOICE vs expected CHAT_MODEL)
```

### Pain Points
1. **Vendor Lock-in**: Limited to OpenAI API only
2. **Inflexible Embeddings**: No way to configure embeddings model or provider
3. **No Custom Base URLs**: Cannot use alternative OpenAI-compatible APIs
4. **Inconsistent Naming**: MODEL_CHOICE should be CHAT_MODEL for clarity
5. **Limited Testing**: Cannot easily test with different providers

## Desired State

### Target Implementation
```yaml
desired_state:
  files:
    - .env.example: Updated with new flexible configuration variables
    - src/utils.py: Modified to support configurable clients for both chat and embeddings
    - All test files: Updated to use new environment variable names
  behavior:
    - Chat model configurable via CHAT_MODEL, CHAT_API_KEY, CHAT_API_BASE
    - Embeddings configurable via EMBEDDINGS_MODEL, EMBEDDINGS_API_KEY, EMBEDDINGS_API_BASE
    - Support for any OpenAI-compatible API (Azure OpenAI, LocalAI, Ollama, etc.)
    - Backward compatibility during transition
    - Clear separation between chat and embeddings configuration
  benefits:
    - Multi-provider support (OpenAI, Azure, LocalAI, Ollama, etc.)
    - Independent configuration of chat and embeddings models
    - Better testing flexibility
    - Clearer configuration naming
    - Future-proof architecture
```

## Implementation Strategy

### Phase 1: Environment Variable Restructuring
1. Update .env.example with new variables
2. Maintain backward compatibility
3. Add clear documentation

### Phase 2: Code Refactoring
1. Create configurable OpenAI clients
2. Update all MODEL_CHOICE references
3. Implement embeddings configuration
4. Add validation and error handling

### Phase 3: Testing and Validation
1. Update test configurations
2. Validate all API calls work with new configuration
3. Test with different provider configurations
4. Performance validation

## Hierarchical Objectives

### High-Level Goal
**Transform fixed OpenAI configuration into flexible multi-provider system**

### Mid-Level Milestones
1. **Environment Configuration Flexibility**: Allow users to configure API keys, base URLs, and models for both chat and embeddings
2. **Code Architecture Modernization**: Refactor utils.py to support configurable OpenAI clients
3. **Backward Compatibility**: Ensure smooth transition without breaking existing deployments
4. **Testing Infrastructure**: Update all tests to use new configuration system

### Low-Level Tasks

#### Task 1: Update Environment Configuration
```yaml
task_name: update_env_configuration
action: MODIFY
file: .env.example
changes: |
  - RENAME MODEL_CHOICE to CHAT_MODEL
  - ADD CHAT_API_KEY (defaults to OPENAI_API_KEY for compatibility)
  - ADD CHAT_API_BASE (defaults to OpenAI base URL)
  - ADD EMBEDDINGS_MODEL (defaults to text-embedding-3-small)
  - ADD EMBEDDINGS_API_KEY (defaults to OPENAI_API_KEY for compatibility)
  - ADD EMBEDDINGS_API_BASE (defaults to OpenAI base URL)
  - ADD comprehensive documentation for each variable
  - PRESERVE existing OPENAI_API_KEY for backward compatibility
validation:
  - command: "grep -E '(CHAT_MODEL|CHAT_API_KEY|CHAT_API_BASE|EMBEDDINGS_MODEL|EMBEDDINGS_API_KEY|EMBEDDINGS_API_BASE)' .env.example"
  - expect: "All new environment variables present with documentation"
```

#### Task 2: Create Configurable OpenAI Clients
```yaml
task_name: create_configurable_clients
action: MODIFY
file: src/utils.py
changes: |
  - CREATE get_chat_client() function that returns configured OpenAI client for chat
  - CREATE get_embeddings_client() function that returns configured OpenAI client for embeddings
  - MODIFY existing openai.chat.completions.create calls to use get_chat_client()
  - MODIFY existing openai.embeddings.create calls to use get_embeddings_client()
  - ADD environment variable reading with fallback logic
  - ADD validation for required configuration
  - PRESERVE backward compatibility with OPENAI_API_KEY
validation:
  - command: "python -c 'from src.utils import get_chat_client, get_embeddings_client; print(\"Clients created successfully\")'"
  - expect: "Functions importable and executable"
```

#### Task 3: Update Model Choice References
```yaml
task_name: update_model_choice_references
action: REPLACE
file: src/utils.py
changes: |
  - REPLACE all os.getenv("MODEL_CHOICE") with os.getenv("CHAT_MODEL", os.getenv("MODEL_CHOICE"))
  - ADD fallback logic to support both old and new variable names
  - UPDATE function documentation to reflect new variable names
validation:
  - command: "grep -n 'MODEL_CHOICE' src/utils.py"
  - expect: "Only fallback references remain, primary usage is CHAT_MODEL"
```

#### Task 4: Update Test Configurations
```yaml
task_name: update_test_configurations
action: MODIFY
file: tests/conftest.py
changes: |
  - UPDATE mock environment to use CHAT_MODEL instead of MODEL_CHOICE
  - ADD CHAT_API_KEY, CHAT_API_BASE mock values
  - ADD EMBEDDINGS_MODEL, EMBEDDINGS_API_KEY, EMBEDDINGS_API_BASE mock values
  - PRESERVE MODEL_CHOICE for backward compatibility testing
validation:
  - command: "python -m pytest tests/conftest.py -v"
  - expect: "All configuration tests pass"
```

#### Task 5: Update Individual Test Files
```yaml
task_name: update_individual_test_files
action: MODIFY
file: tests/test_mcp_basic.py
changes: |
  - REPLACE os.environ.setdefault("MODEL_CHOICE", ...) with CHAT_MODEL
  - ADD environment setup for new variables
  - PRESERVE fallback testing for MODEL_CHOICE
validation:
  - command: "python -m pytest tests/test_mcp_basic.py -v"
  - expect: "All tests pass with new configuration"
```

#### Task 6: Update Additional Test Files
```yaml
task_name: update_additional_test_files
action: MODIFY
file: tests/test_mcp_server.py
changes: |
  - REPLACE os.environ.setdefault("MODEL_CHOICE", ...) with CHAT_MODEL
  - ADD environment setup for new variables
  - PRESERVE fallback testing for MODEL_CHOICE
validation:
  - command: "python -m pytest tests/test_mcp_server.py -v"
  - expect: "All tests pass with new configuration"
```

#### Task 7: Update Knowledge Graph Scripts
```yaml
task_name: update_knowledge_graph_scripts
action: MODIFY
file: knowledge_graphs/test_script.py
changes: |
  - REPLACE os.getenv('MODEL_CHOICE', 'gpt-4.1-mini') with os.getenv('CHAT_MODEL', os.getenv('MODEL_CHOICE', 'gpt-4.1-mini'))
  - ADD support for new environment variables
validation:
  - command: "python knowledge_graphs/test_script.py --help"
  - expect: "Script loads without error"
```

#### Task 8: Update README Documentation
```yaml
task_name: update_readme_documentation
action: MODIFY
file: README.md
changes: |
  - UPDATE configuration section to document new environment variables
  - ADD examples for different provider configurations (Azure OpenAI, LocalAI)
  - UPDATE MODEL_CHOICE references to CHAT_MODEL
  - ADD migration guide for existing users
  - PRESERVE examples showing backward compatibility
validation:
  - command: "grep -E '(CHAT_MODEL|EMBEDDINGS_MODEL)' README.md"
  - expect: "Documentation updated with new variable names"
```

#### Task 9: Add Client Configuration Validation
```yaml
task_name: add_client_configuration_validation
action: ADD
file: src/utils.py
changes: |
  - CREATE validate_chat_config() function
  - CREATE validate_embeddings_config() function
  - ADD helpful error messages for missing configuration
  - ADD warnings for deprecated MODEL_CHOICE usage
  - ADD debug logging for configuration loading
validation:
  - command: "python -c 'import os; os.environ.clear(); from src.utils import validate_chat_config'"
  - expect: "Validation functions work and provide clear error messages"
```

#### Task 10: Integration Testing
```yaml
task_name: integration_testing
action: CREATE
file: tests/test_flexible_api_config.py
changes: |
  - CREATE comprehensive tests for new configuration system
  - ADD tests for backward compatibility
  - ADD tests for different provider configurations
  - ADD tests for error handling and validation
  - ADD performance tests to ensure no regression
validation:
  - command: "python -m pytest tests/test_flexible_api_config.py -v"
  - expect: "All integration tests pass"
```

## Risk Assessment & Mitigation

### High Risk Items
1. **Breaking Existing Deployments**
   - **Mitigation**: Comprehensive backward compatibility with MODEL_CHOICE and OPENAI_API_KEY
   - **Rollback**: Keep original environment variable support indefinitely

2. **API Configuration Errors**
   - **Mitigation**: Extensive validation and clear error messages
   - **Testing**: Test with multiple provider configurations

3. **Performance Impact**
   - **Mitigation**: Lazy client initialization and caching
   - **Validation**: Performance benchmarks before/after

### Medium Risk Items
1. **Test Suite Stability**
   - **Mitigation**: Gradual test migration with parallel old/new support
   - **Validation**: Run full test suite after each change

2. **Documentation Gaps**
   - **Mitigation**: Comprehensive documentation updates
   - **Examples**: Multiple provider configuration examples

## Migration Strategy

### For Existing Users
1. **No Immediate Action Required**: Existing MODEL_CHOICE and OPENAI_API_KEY continue to work
2. **Optional Migration**: Users can gradually adopt new variable names
3. **Enhanced Functionality**: New users get full flexibility immediately

### Deprecation Timeline
1. **Phase 1 (Current)**: Full backward compatibility
2. **Phase 2 (Future)**: Deprecation warnings for MODEL_CHOICE
3. **Phase 3 (Long-term)**: Remove MODEL_CHOICE support (major version)

## Success Criteria

### Functional Requirements
- [ ] Users can configure chat model via CHAT_MODEL, CHAT_API_KEY, CHAT_API_BASE
- [ ] Users can configure embeddings via EMBEDDINGS_MODEL, EMBEDDINGS_API_KEY, EMBEDDINGS_API_BASE
- [ ] Backward compatibility with MODEL_CHOICE and OPENAI_API_KEY maintained
- [ ] Support for Azure OpenAI, LocalAI, and other OpenAI-compatible APIs
- [ ] Clear error messages for configuration issues

### Quality Requirements
- [ ] All existing tests continue to pass
- [ ] No performance regression
- [ ] Clear documentation for new configuration
- [ ] Migration guide for existing users
- [ ] Comprehensive error handling

### Integration Requirements
- [ ] Works with existing Docker setup
- [ ] Compatible with current MCP server architecture
- [ ] Supports all existing RAG strategies
- [ ] Maintains current security practices

## Dependencies

### Internal Dependencies
- All files using MODEL_CHOICE environment variable
- Test suite configuration
- Documentation updates

### External Dependencies
- OpenAI Python client library (already present)
- Environment variable loading (dotenv)
- Existing Docker configuration

### Sequence Dependencies
1. Environment variable updates must come first
2. Code refactoring depends on new environment structure
3. Testing updates require code changes to be complete
4. Documentation should be updated throughout the process

## Testing Strategy

### Unit Tests
- Test each new configuration function individually
- Test backward compatibility scenarios
- Test error handling and validation

### Integration Tests
- Test with real API providers (using test keys)
- Test Docker deployment with new configuration
- Test MCP server functionality end-to-end

### Performance Tests
- Benchmark API call performance before/after
- Validate no significant overhead from new configuration layer
- Test concurrent API calls with new client management

## Rollback Plan

### Immediate Rollback
- Keep all original environment variable support
- Maintain original code paths as fallbacks
- Version control allows immediate reversion

### Gradual Rollback
- Disable new configuration features via feature flags
- Redirect all calls to original implementation
- Preserve original error handling

## Future Enhancements

### Short-term Possibilities
- Configuration validation CLI tool
- Provider-specific optimization settings
- Automatic provider detection

### Long-term Vision
- Plugin architecture for different providers
- Advanced retry and failover strategies
- Cost optimization across providers
- Multi-provider load balancing

---

**Implementation Timeline**: 2-3 development cycles
**Risk Level**: Medium (mitigated by backward compatibility)
**Dependencies**: Low (mostly internal refactoring)
**User Impact**: High positive (increased flexibility, no breaking changes)