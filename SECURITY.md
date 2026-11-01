# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

The **SKETCH2SHAPE AI** team takes software security seriously. If you discover a security vulnerability or sensitive information exposure within this repository, please do not disclose it via a public GitHub issue.

Instead, please report the vulnerability confidentially:
- **Email**: Open a security advisory on GitHub or email the maintainer directly.
- **Response Time**: We acknowledge receipt of vulnerability reports within 48 hours and aim to provide a remediation plan or fix within 7 business days.

Please include the following details in your report:
1. Description of the vulnerability or security weakness.
2. Steps to reproduce the issue (including any sample payloads, curl commands, or script snippets).
3. The potential impact of the vulnerability.
4. Proposed fix or mitigation (if available).

---

## Environment Variables & Secret Management

- **Zero Secrets in Repository**: This repository strictly ignores `.env` and all credential files via `.gitignore`.
- **Pre-Commit Checks**: Contributors must never commit real API keys (such as `GROQ_API_KEY`, `TRIPO_API_KEY`, or `MESHY_API_KEY`).
- **Template Configuration**: Always copy `.env.example` to `.env` locally and populate it with your own private keys. Never commit your populated `.env`.
- **Rotated Credentials**: Any API key accidentally pushed to a remote branch must be immediately revoked and rotated at the provider dashboard.
