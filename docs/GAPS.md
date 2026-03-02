# Documentation Gaps

This document is a post-overhaul gap analysis for the LLM Council project. After the comprehensive documentation overhaul (README rewrite, module/function docstring audit, example files, CONTRIBUTING.md update), the items below remain missing or incomplete. Use this as a roadmap for future documentation work.

---

## 1. CHANGELOG.md

**What's missing:** The project has no changelog tracking version history, breaking changes, new features, deprecations, or bug fixes across releases.

**Why it matters:** Without a changelog, users upgrading between versions have no way to know what changed, what broke, or what new capabilities are available. Contributors also lack a record of project evolution, making it harder to understand when and why features were introduced.

**Recommended action:** Create a `CHANGELOG.md` in the project root following the [Keep a Changelog](https://keepachangelog.com) format. Each release entry should use these categories:

- **Added** — new features
- **Changed** — changes to existing functionality
- **Deprecated** — features that will be removed in upcoming releases
- **Removed** — features removed in this release
- **Fixed** — bug fixes
- **Security** — vulnerability patches

Example structure:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Streaming output support (`--stream` flag)
- Response caching with SQLite backend
```

---

## 2. Migration Guide

**What's missing:** There is no migration guide documenting how to upgrade between versions when breaking changes are introduced (e.g., config schema changes, CLI flag renames, provider API updates).

**Why it matters:** As the project evolves, config formats, CLI flags, and provider interfaces may change. Without a migration guide, users are left to discover breaking changes through runtime errors. This is especially important for skill users whose `council-config.json` files may become incompatible after an upgrade.

**Recommended action:** Create a `docs/MIGRATION.md` when the first breaking change is introduced. Structure it by version:

```markdown
# Migration Guide

## Upgrading to v2.0

### Config changes
- `budget_limit` renamed to `max_cost_usd` in the `budget` section
- `cache_ttl` now accepts seconds (previously minutes)

### CLI changes
- `--no-stream` removed; streaming is now opt-in via `--stream`
```

Until breaking changes occur, a placeholder note in the README or CHANGELOG is sufficient.

---

## 3. Auto-Generated API Reference

**What's missing:** The project relies entirely on in-source docstrings for API documentation. There is no auto-generated, browsable API reference hosted as HTML or served via a documentation site.

**Why it matters:** While in-source docstrings are valuable for contributors reading code directly, they are not easily discoverable for users who want to browse the full API surface without cloning the repo. A hosted API reference also enables search, cross-linking, and a more structured reading experience.

**Recommended action:** Set up [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) with [mkdocstrings](https://mkdocstrings.github.io/) to auto-generate API docs from the existing Google-style docstrings. This is the lighter-weight option compared to Sphinx and integrates well with GitHub Pages.

Minimal `mkdocs.yml` to get started:

```yaml
site_name: LLM Council API Reference
theme:
  name: material

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            members_order: source

nav:
  - Home: index.md
  - API Reference:
    - Council: api/council.md
    - Config: api/models.md
    - Providers: api/providers.md
    - Cache: api/cache.md
    - Budget: api/budget.md
    - Security: api/security.md
    - Flattener: api/flattener.md
    - Parsing: api/parsing.md
```

Install with:

```bash
uv add --dev mkdocs-material mkdocstrings[python]
```

Alternatively, [Sphinx](https://www.sphinx-doc.org/) with `autodoc` and the `napoleon` extension (for Google-style docstrings) is a more established option if richer cross-referencing or PDF output is needed.

---

## 4. Skill Integration Documentation Gaps

**What's missing:** While the README now documents skill setup and invocation for both Claude Code and OpenClaw, several areas remain under-documented:

- **Skill versioning and updates:** Neither `.claude/commands/council.md` nor `skills/council/SKILL.md` document how users should handle skill updates when the upstream project changes. There is no guidance on re-syncing skill files after a `git pull`.
- **Skill-specific troubleshooting:** The skill files document basic error handling (model failure, missing config, missing credentials) but lack detailed troubleshooting steps — e.g., what to do when the skill script fails silently, how to enable verbose output from within a skill invocation, or how to debug `{baseDir}` path resolution issues in OpenClaw.
- **Shared script divergence:** The README documents that `.claude/skills/council/scripts/council.py` and `skills/council/scripts/council.py` share the same underlying script, but there is no mechanism or documentation for keeping them in sync (e.g., a symlink strategy or a CI check).
- ~~**OpenRouter support in skills:** The skill files (`.claude/commands/council.md`, `skills/council/SKILL.md`) document Bedrock and Poe models but do not mention OpenRouter as a provider option in the config mode instructions, even though the core library supports it.~~ **RESOLVED** — OpenRouter added to both skill files with model examples, config examples, and parameter tables.
- **Plugin metadata:** `.claude/skills/council/.claude-plugin/` contains `marketplace.json` and `plugin.json` but these files are not documented anywhere — their purpose, schema, and how to update them is unclear.

**Why it matters:** Skill-based usage is the primary interaction mode for most users. Gaps in skill documentation directly impact the majority of the user base.

**Recommended action:**
- Add a "Updating Skills" section to the README's Skill Usage documentation explaining how to pull updates and re-sync skill files.
- Add OpenRouter model examples to both skill command files' config mode sections.
- Document the `.claude-plugin/` metadata files in CONTRIBUTING.md or a dedicated skill development guide.
- Consider adding a `--verbose` passthrough flag to the skill scripts for debugging.

---

## ~~5. Troubleshooting Guide~~ **RESOLVED**

**RESOLVED** — Created `docs/TROUBLESHOOTING.md` covering: missing API keys, AWS credentials, model not found, budget exceeded, timeout errors, stale cache, invalid config, ranking parse failures, circuit breaker behavior, and debugging tips.

~~**What's missing:** There is no dedicated troubleshooting guide covering common errors users encounter when running the council.~~

**Why it matters:** Users hitting errors often abandon tools rather than debugging them. A troubleshooting guide with clear symptoms, causes, and fixes dramatically reduces support burden and improves the onboarding experience.

**Recommended action:** Create a `docs/TROUBLESHOOTING.md` covering at minimum these common error scenarios:

| Error | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| Missing API keys | `POE_API_KEY not set` or `OPENROUTER_API_KEY not set` | Environment variable not configured | Set the variable in `.env` or shell profile; see README env var section |
| AWS credentials | `NoCredentialsError` or `ExpiredTokenError` | AWS CLI not configured or session expired | Run `aws configure` or refresh SSO with `aws sso login` |
| Model not found | `Model 'xyz' not found` or `ValidationException` | Typo in `model_id`, model not available in region, or bot name changed on Poe | Verify model ID against provider docs; use `--list-models` to see available models |
| Budget exceeded | `BudgetExhaustedError` or council stops mid-run | `max_cost_usd` or `max_tokens` limit reached | Increase budget limits in config or remove the `budget` section |
| Timeout errors | `asyncio.TimeoutError` or `soft_timeout exceeded` | Model taking too long to respond, network issues | Increase `soft_timeout` in config; check network connectivity; the circuit breaker will skip slow models automatically |
| Cache issues | Stale responses, unexpected results | Cached responses from a previous config | Run `llm-council --clear-cache` or use `--no-cache` for a fresh run |
| Invalid config | `ValidationError` on startup | Malformed `council-config.json` | Validate JSON syntax; check field names against the Configuration Reference in the README |
| Ranking parse failures | `No valid ballots` or low ballot count | Models returning rankings in unexpected format | Check `--log-dir` output for raw model responses; increase `stage2_retries` in config |

---

## 6. Priority Recommendations

Address these gaps in the following order:

1. **Troubleshooting guide** (high impact, low effort) — Directly reduces user friction. Most of the content can be derived from existing error handling code and common support questions.

2. **CHANGELOG.md** (high impact, low effort) — Start tracking changes now, even retroactively for the current version. This becomes increasingly valuable with every release.

3. **Skill integration gaps** (medium impact, medium effort) — Fill in the OpenRouter omission in skill files and add update/sync guidance. These affect the primary user path.

4. **Auto-generated API docs** (medium impact, medium effort) — Set up mkdocs-material with mkdocstrings. The Google-style docstrings from this overhaul are already in place, so the content exists — it just needs a build pipeline.

5. **Migration guide** (low urgency, create when needed) — Not needed until the first breaking change. Add a placeholder note in the CHANGELOG when the time comes.
