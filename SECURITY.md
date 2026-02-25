# Security Policy

## Responsible Disclosure

If you discover a security vulnerability, please email security issues to the maintainer rather than opening a public issue. We appreciate your efforts to responsibly disclose your findings.

## Security Features

### Injection Hardening
- **Anonymized peer review** - Models evaluate "Response A/B/C" without knowing which model produced each response, preventing bias-based manipulation
- **Fenced response wrapping** - Model responses are wrapped in code blocks during peer review to prevent prompt injection
- **System message protection** - Ranking and synthesis models are instructed to ignore manipulation attempts in responses
- **Nonce-based XML wrapping** - Unique nonces prevent response boundary confusion attacks
- **Input sanitization** - User inputs are sanitized to remove potentially malicious patterns

### Budget Controls
- **Token limits** - Configurable token budgets prevent runaway costs
- **Provider rate limiting** - Built-in retry logic with exponential backoff
- **Graceful degradation** - If a model fails, the council continues with remaining models

## Threat Model

### Mitigated Threats
- **Prompt injection** - Multiple layers of defense prevent models from manipulating the council process
- **Model bias** - Anonymization prevents models from favoring their own family or disfavoring competitors
- **Response boundary confusion** - XML wrapping with nonces ensures clear response boundaries
- **Ranking manipulation** - System prompts explicitly instruct models to ignore manipulation attempts

### Out of Scope
- **API key security** - Users are responsible for securing their own API keys
- **Network security** - TLS is handled by the underlying HTTP libraries
- **Supply chain attacks** - Dependencies are managed through standard Python tooling

## API Key Best Practices

### Storage
- Never commit API keys to version control
- Use environment variables or secure vaults
- Rotate keys regularly

### Environment Variables
```bash
# AWS credentials for Bedrock
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_SESSION_TOKEN=your-token  # if using temporary credentials

# Poe.com API key
export POE_API_KEY=your-poe-key
```

### Configuration Files
- The `council-config.json` file should not contain API keys
- Keep configuration files with sensitive data in `.gitignore`

## Dependencies

We recommend running security audits on dependencies:

```bash
# Install pip-audit
pip install pip-audit

# Run audit
pip-audit

# Or with uv
uv pip audit
```

## Reporting

For security concerns or questions, please contact the maintainer through GitHub issues (for non-sensitive matters) or via email for sensitive security disclosures.