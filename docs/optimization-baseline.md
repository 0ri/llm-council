# Optimization Baseline

Captured 2026-03-03 before optimization work begins.

## Startup Time

```
$ time llm-council --help
real    0.376s
user    0.16s
sys     0.06s
```

## Call Counts (per full run)

| Function             | Call sites | Notes                              |
|---------------------|------------|------------------------------------|
| `coerce_model_config` | 13 sites   | 1 def + 6 stages + 6 providers    |
| `estimate_tokens`     | 4 sites    | 1 def + 2 stages + 1 cost tracker |

## Import Time

Measured with `python -X importtime -c "import llm_council"`.
Eagerly imports all providers (`bedrock`, `openrouter`, `poe`) plus their transitive dependencies (`boto3`, `httpx`, `fastapi_poe`).

## Test Suite

```
473 passed, 7 warnings in 32.50s
```

## File Sizes

| File                        | LOC |
|-----------------------------|-----|
| `stages.py`                 | 915 |
| `council.py`                | 439 |
| `budget.py`                 | 378 |
| `cache.py`                  | 328 |
| `openrouter.py`             | 326 |
| `parsing.py`                | 316 |
| `progress.py`               | 298 |
| `providers/__init__.py`     | 278 |
| `bedrock.py`                | 268 |
| `poe.py`                    | 239 |
| `models.py`                 | 241 |
| `cost.py`                   | 206 |
| `context.py`                | 191 |
| `formatting.py`             | 186 |
| `security.py`               | 176 |
| `persistence.py`            | 93  |
| `cli.py`                    | ~300|

## Golden Outputs

Deterministic golden output tests in `tests/test_golden_outputs.py` cover:
1. Stage 1-only formatting structure
2. Full 3-stage non-streaming output structure
3. Stream fallback behavior
4. Strict-ballot failure mode
5. Stage 2 output structure (rankings + individual responses)

These tests verify output structure is preserved through refactoring.

---

## Post-Optimization Metrics (2026-03-03)

### Startup Time

```
$ time llm-council --help
real    0.285s (24% faster)
```

### Test Suite

```
493 passed, 7 warnings in 32.07s  (+20 tests)
```

### Call Counts

| Function             | Call sites | Change                             |
|---------------------|------------|------------------------------------|
| `coerce_model_config` | 7 sites    | Removed from 6 provider methods    |
| `estimate_tokens`     | 2 sites    | Unified in execution helpers       |

### File Sizes — stages/ package (was stages.py: 915 LOC)

| File                        | LOC |
|-----------------------------|-----|
| `stages/__init__.py`        |  27 |
| `stages/execution.py`       | 378 |
| `stages/stage1.py`          | 117 |
| `stages/stage2.py`          | 310 |
| `stages/stage3.py`          | 160 |
| **Max module size**         | **378** (was 915, 59% reduction) |

### Key Changes

| Provider file       | LOC (before → after) |
|---------------------|---------------------|
| `poe.py`            | 239 → 211           |
| `openrouter.py`     | 326 → 321           |
| `bedrock.py`        | 268 → 263           |
| `persistence.py`    | 93 → 139 (buffered) |

### New Files

| File                          | LOC | Purpose                              |
|-------------------------------|-----|--------------------------------------|
| `run_options.py`              |  30 | RunOptions dataclass                 |
| `test_execution_invariants.py`| 210 | Budget/CB lifecycle invariant tests  |
| `test_golden_outputs.py`      | 125 | Deterministic output regression tests|

### Summary

- 18 commits total (Commits 0-12, 13-18)
- Monolithic `stages.py` (915 LOC) split into 5-module package (max 378 LOC)
- Lazy provider imports via explicit module registry
- Budget lifecycle centralized in `_managed_execution` context manager
- `coerce_model_config` removed from 6 provider call sites
- `RunOptions` dataclass consolidates 9 `run_council` parameters
- Buffered JSONL persistence with per-stage flush
- All 493 tests pass, all 7 golden output tests pass

---

## Tier 1–2 Optimization (2026-03-04)

Addressed findings from a multi-model council audit of the codebase.

### Bug Fixes

- **Config duality bug (P0):** `config.get()` was called on a Pydantic `CouncilConfig` object in `run_council()`, which would raise `AttributeError` if a caller passed a typed config directly. Config is now parsed once early into `validated: CouncilConfig` with typed attribute access throughout.
- **Thread-local connection leak:** `ResponseCache.close()` only closed the main connection but not thread-local connections created via `_get_thread_conn()`. Added `_thread_conns` registry with lock-protected cleanup.

### Structural Changes

| Change | Files | Impact |
|--------|-------|--------|
| Parse config once, use typed access | `council.py`, `budget.py` | Eliminated 9 `config.get()` calls after parse |
| `defaults.py` constants module | New file + 4 importers | Unified `DEFAULT_CACHE_TTL`, `DEFAULT_SOFT_TIMEOUT`, `DEFAULT_STAGE2_RETRIES` |
| `resolve_template()` helper | `prompts.py`, `stage2.py`, `stage3.py` | De-duplicated prompt template resolution |
| `_build_request_body()` in Bedrock | `bedrock.py` | De-duplicated body construction between `query()` and `astream()` |
| Cache `_do_get`/`_do_put` helpers | `cache.py` | De-duplicated sync/thread-local get/put logic |
| `_SCHEMA_SQL` constant | `cache.py` | Eliminated duplicated CREATE TABLE statement |
| `RunManifest.create(run_id=...)` | `manifest.py`, `council.py` | Eliminated post-hoc `manifest.run_id = run_id` |
| Shared test fixtures | `conftest.py` + 7 test files | Replaced 7 local `_make_ctx`/`_make_ctx_factory` factories |
| Cache fault logging | `stage1.py` | `logger.warning` → `logger.exception` for full traceback |
| Deprecate `calculate_borda_score` | `aggregation.py` | `DeprecationWarning` on unused function |

### File Sizes (changed files)

| File | LOC (before → after) |
|------|---------------------|
| `council.py` | 456 → 451 |
| `budget.py` | 378 → 394 |
| `cache.py` | 328 → 342 |
| `prompts.py` | 82 → 107 |
| `bedrock.py` | 263 → 259 |
| `aggregation.py` | 185 → 195 |
| `manifest.py` | 104 → 109 |
| `defaults.py` | *(new)* → 14 |

### Test File Sizes (changed files)

| File | LOC (before → after) |
|------|---------------------|
| `conftest.py` | 67 → 122 |
| `test_council_integration.py` | 408 → 386 |
| `test_golden_outputs.py` | 298 → 271 |
| `test_stages.py` | 484 → 461 |
| `test_stream_model.py` | 268 → 265 |
| `test_streaming_integration.py` | 244 → 227 |
| `test_audit_fixes.py` | 367 → 349 |
| `test_improvements.py` | 323 → 311 |
| **Test total (changed)** | **2,466 → 2,392 (−74)** |

### Summary

- 1 commit, 22 files changed (+367 −362, net −9 LOC)
- Fixed latent P0 correctness bug (config duality)
- Fixed thread-local SQLite connection leak
- Source +79 LOC (new helpers, typed paths, deprecation), tests −74 LOC (shared fixtures)
- All 493 tests pass, ruff clean
