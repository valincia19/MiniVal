# Security Policy

The MiniVal team takes the security of our software, dependencies, and users seriously.

---

## Supported Versions

MiniVal is an evolving project. Security fixes and patches are applied to the `main` branch:

| Version | Supported |
| :--- | :--- |
| `main` (latest) | :white_check_mark: |
| older commits | :x: |

---

## Scope & Considerations

MiniVal includes both standalone LLM training code and an OpenAI-compatible FastAPI server (`scripts/serve_api.py`).

Areas of security relevance include:
- **API Server & Endpoints**: Authentication mechanisms (e.g. `MINIVAL_API_KEY`), denial of service via uncontrolled generation lengths, or unsafe request handling.
- **Model Deserialization**: Safe loading of checkpoint formats (preferring `safetensors` over untrusted `.pth` / `.pt` pickle binaries).
- **Dependency Vulnerabilities**: Exploitable flaws in upstream AI/ML dependencies.

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, please follow responsible disclosure practices:

1. **GitHub Private Vulnerability Advisory (Recommended)**:
   - Navigate to the repository's [Security Advisories](https://github.com/valincia19/MiniVal/security/advisories/new) page.
   - Submit the details privately to the project maintainers.

2. **Direct Maintainer Contact**:
   - Contact the repository maintainer directly via GitHub profile: [@valincia19](https://github.com/valincia19).
   - *(Maintainer note: To configure a dedicated security email address, add your security contact email to this section or enable GitHub Private Vulnerability Reporting in repository Settings > Code security and analysis).*

### When reporting, please include:
- A description of the vulnerability and its potential impact.
- Clear reproduction steps or a minimal proof-of-concept.
- Any suggested mitigations or patches.

### Response Timeline:
- The maintainers will acknowledge receipt within 48-72 hours.
- A private patch will be prepared and tested before publishing an advisory or releasing a fix.
