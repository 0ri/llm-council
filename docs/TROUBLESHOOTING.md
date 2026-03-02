# Troubleshooting

Common errors and how to fix them.

## Missing API Keys

**Symptom:** `OPENROUTER_API_KEY not set` or `POE_API_KEY not set`

**Cause:** The environment variable for your provider isn't configured.

**Fix:** Set the variable in your shell profile or `.env` file:

```bash
export OPENROUTER_API_KEY=your-key   # Default config uses OpenRouter
export POE_API_KEY=your-key          # Only if using Poe provider
```

You only need keys for providers in your `council-config.json`. The default config requires only `OPENROUTER_API_KEY`.

## AWS Credentials

**Symptom:** `NoCredentialsError` or `ExpiredTokenError` when using Bedrock models.

**Cause:** AWS CLI not configured or session expired.

**Fix:**

```bash
aws configure           # Set up credentials
aws sso login           # Or refresh SSO session
```

## Model Not Found

**Symptom:** `Model 'xyz' not found`, `ValidationException`, or `404` from OpenRouter.

**Cause:** Typo in `model_id`, model not available in your region (Bedrock), or bot name changed (Poe).

**Fix:**
- Run `llm-council --list-models` to see available models
- For Bedrock: verify the model is enabled in your AWS region
- For OpenRouter: check the model ID at [openrouter.ai/models](https://openrouter.ai/models)
- For Poe: verify the bot name on poe.com

## Budget Exceeded

**Symptom:** `BudgetExceededError` or council stops mid-run.

**Cause:** The `max_cost_usd` or `max_tokens` limit in the `budget` config section was reached.

**Fix:** Increase the limits or remove the `budget` section entirely:

```json
{
  "budget": {
    "max_tokens": 500000,
    "max_cost_usd": 5.0
  }
}
```

To disable budget limits, omit the `budget` key from your config.

## Timeout Errors

**Symptom:** `asyncio.TimeoutError` or models returning no response.

**Cause:** A model is taking too long; network issues.

**Fix:**
- Increase `soft_timeout` in your config (default: 300 seconds)
- The circuit breaker automatically skips models that fail 3 times consecutively (see [Resilience](#resilience-circuit-breakers-and-retries) below)
- Set `min_responses` in config to proceed with partial results before all models respond

## Stale Cache Results

**Symptom:** Getting old responses even after changing your question or config.

**Cause:** Cached Stage 1 responses from a previous run.

**Fix:**

```bash
llm-council --clear-cache          # Delete all cached responses
llm-council --no-cache "question"  # Bypass cache for one run
llm-council --cache-ttl 0 "question"  # Bypass reads, still writes new entries
```

Check cache status with `llm-council --cache-stats`.

## Invalid Config

**Symptom:** `ValidationError` on startup with a list of field errors.

**Cause:** Malformed JSON or invalid field values in `council-config.json`.

**Fix:**
- Validate JSON syntax (trailing commas, missing quotes)
- Check field names against the [Configuration Reference](../README.md#configuration-reference)
- Ensure `council_models` has at least one entry
- Ensure `budget_tokens` (Bedrock) is between 1024 and 128000

## Ranking Parse Failures

**Symptom:** `No valid ballots` warning, low ballot count in output, or unexpected rankings.

**Cause:** Models returning rankings in an unparseable format.

**Fix:**
- Increase `stage2_retries` in config (default: 1, max: 5) to give models more attempts
- Enable `strict_ballots` in config to reject partial or ambiguous rankings
- Check raw model responses with `--log-dir ./logs` and inspect the JSONL files
- Run with `-v` for verbose logging to see ballot parsing details

## Resilience: Circuit Breakers and Retries

LLM Council includes automatic resilience mechanisms:

- **Circuit breaker:** After 3 consecutive failures from a model, that model is skipped for 60 seconds before retrying. This prevents wasting time and tokens on a consistently failing provider.
- **Soft timeout:** If `soft_timeout` expires and at least `min_responses` models have responded, the pipeline proceeds without waiting for stragglers.
- **Stage 2 retries:** Invalid ranking ballots are retried up to `stage2_retries` times (default: 1).
- **Graceful degradation:** If a model fails entirely, the council continues with the remaining models rather than failing the whole run.

## Debugging Tips

- Use `-v` (verbose) to see DEBUG-level logs on stderr
- Use `--manifest` to see the full run manifest (config hash, timestamps, model list) on stderr
- Use `--log-dir ./logs` to persist complete JSONL logs for post-mortem analysis
- Use `--dry-run` to preview config and estimated API calls without spending tokens
- Use `--stage 1` to test just the response collection stage
