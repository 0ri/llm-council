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
