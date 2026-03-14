# Archived Files

This directory contains files that are no longer used in the active codebase.

## Files Archived

### `vocab.py` (463 lines)
- **Status**: Completely unused
- **Reason**: Not imported anywhere in the codebase
- **Date Archived**: 2024-11-05
- **Notes**: Vocabulary enhancement functionality was removed as part of simplification efforts. The file is kept for reference but is not part of the active codebase.

### `debug_system.py` (337 lines)
- **Status**: Completely unused
- **Reason**: Not imported anywhere in the codebase
- **Date Archived**: 2024-11-05
- **Notes**: Debug system functionality may have been replaced or integrated elsewhere. Kept for reference.

### `orchestrator.py` (554 lines)
- **Status**: Broken (missing dependencies)
- **Reason**: Imports modules that don't exist:
  - `intent_classifier.py`
  - `llm_intent_classifier.py`
  - `entity_linker.py`
  - `graph_retrieval.py`
- **Date Archived**: 2024-11-05
- **Notes**: This file was already commented out in `cli.py` (line 25). The `CLAIREOrchestrator` class has been replaced by `LangChainOrchestrator` and `LLMOrchestrator`.

## Total Archived Code
- **3 files**
- **~1,354 lines of code**

## Restoration
If you need to restore any of these files:
1. Move the file back to `src/claire_kg/`
2. Fix any missing dependencies
3. Update imports in files that use them
4. Test thoroughly before committing

