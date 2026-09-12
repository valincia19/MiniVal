# Contributing to MiniVal

Thank you for your interest in contributing to **MiniVal**! ⚡

MiniVal is an educational, lightweight, from-scratch Large Language Model (LLM) implementation designed to make modern generative AI transparent and accessible—from basic pretraining and instruction tuning (SFT) to advanced techniques like Mixture-of-Experts (MoE), LoRA adaptation, Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO reasoning RL ala DeepSeek-R1).

Our goal is to keep the codebase clean, highly readable, modular, and easy to run on both developer laptops and GPU workstations.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Running Verification & Tests](#running-verification--tests)
- [Branch & Commit Conventions](#branch--commit-conventions)
- [Pull Request Workflow](#pull-request-workflow)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Enhancements](#proposing-enhancements)
- [Documentation Improvements](#documentation-improvements)
- [Code Style & Conventions](#code-style--conventions)
- [Review & Maintainer Expectations](#review--maintainer-expectations)

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful, considerate, and constructive in all discussions and reviews.

---

## Prerequisites

- **Python**: 3.10 or higher (Python 3.11 recommended)
- **Git**: 2.25+
- **Hardware**:
  - **CPU-only**: Fully supported! You can run tests, export models, and experiment with the `tiny` preset without a GPU.
  - **NVIDIA GPU** (Optional): CUDA 11.8+ / 12.x for accelerated training and VRAM-efficient kernels (SDPA / bfloat16).

---

## Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/MiniVal.git
   cd MiniVal
   ```

2. **Create and Activate a Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Your Environment**:
   ```bash
   python minival.py info
   ```
   This displays Python version, PyTorch version, CUDA availability, and device VRAM specs.

---

## Project Architecture

MiniVal follows a minimal, modular design philosophy:

| Component | Path | Description |
| :--- | :--- | :--- |
| **Unified CLI** | `minival.py` | Single entry point for chat, training, serving, and diagnostics |
| **Configuration** | `config.py` | Model architectures (`tiny`, `base`, `pro`, `moe`) and training configs |
| **Training Engine** | `engine.py` | Implementation of Pretraining, SFT, LoRA, DPO, GRPO, and Distillation |
| **Rollout Engine** | `rollout.py` | Trajectory generation and log-probability calculation for RL |
| **Reward Engine** | `rewards.py` | Modular rule-based rewards (format, tool-calling, thinking length) |
| **Model Core** | `model/model_minival.py` | Transformer architecture with GQA, RoPE/YaRN, and MoE |
| **LoRA Engine** | `model/model_lora.py` | Custom low-rank adapter injection, freezing, and merging |
| **Dataset Loader** | `dataset/universal.py` | Universal stream loader for `.txt` and conversational `.jsonl` |
| **Scripts & Tools** | `scripts/` | Streamlit web demo, OpenAI-compatible FastAPI server, eval benchmarks |
| **Regression Suite** | `test_verification.py` | Full 7-suite verification script |

---

## Running Verification & Tests

Before submitting any code changes, **always run the regression verification suite**:

```bash
python test_verification.py
```

This validates:
1. **Config & Presets**: Hyperparameter validation across all stages and model sizes.
2. **Model & MoE Aux Loss**: Forward passes, tensor dimensions, and routing auxiliary loss.
3. **LoRA**: Parameter freezing, adapter attachment, forward pass, save/load, and weight merging.
4. **Rewards Engine**: Format rewards, tool-call detection, and composite scoring.
5. **UniversalDataset**: Subsequence token masking and loss target alignment.
6. **Rollout Engine**: Per-token log-probability extraction for RL loops.
7. **API Schemas**: FastAPI / Pydantic contract compliance for OpenAI endpoints.

> [!TIP]
> If you have `pytest` installed in your environment, you can also run:
> ```bash
> pytest test_verification.py
> ```

All suites must pass without errors or regressions.

---

## Branch & Commit Conventions

### Branch Naming
Create focused branches using descriptive prefixes:
- `feat/<feature-name>`: New capabilities or modules (e.g., `feat/kv-cache-paged`)
- `fix/<bug-name>`: Bug fixes (e.g., `fix/grpo-padding-mask`)
- `docs/<topic>`: Documentation updates or tutorials (e.g., `docs/install-guide`)
- `perf/<optimization>`: Performance or memory improvements (e.g., `perf/flash-attn-sdpa`)
- `test/<suite>`: Adding or refining test coverage (e.g., `test/reward-unit-tests`)

### Commit Messages
Write clear, concise commit messages in the imperative mood. Following [Conventional Commits](https://www.conventionalcommits.org/) is recommended:
```text
feat(rewards): add regex reward for math boxed format
fix(engine): ensure left-padding for generation rollouts
docs(readme): clarify CPU training flags
```

---

## Pull Request Workflow

1. **Keep Pull Requests Focused**: A PR should address one specific problem or feature. Avoid bundling unrelated refactors or formatting changes.
2. **Sync with Main**: Keep your branch up to date with `origin/main` before opening a PR:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
3. **Run Tests Locally**: Ensure `python test_verification.py` passes cleanly.
4. **Do Not Commit Large Weights or Datasets**:
   - Model weights (`*.safetensors`, `*.pth`, `*.bin`, `*.gguf`) and large dataset files (`*.jsonl`, `*.txt`) belong in `.gitignore`.
   - Never commit binaries or credentials to git history.
5. **Open a Pull Request**:
   - Use the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
   - Describe the motivation, changes made, and testing proof.

---

## Reporting Bugs

If you discover a bug or unintended behavior:
1. Check the [issue tracker](https://github.com/valincia19/MiniVal/issues) to ensure it has not already been reported.
2. Open a new issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
3. Provide:
   - A clear, concise title.
   - Exact steps to reproduce the issue.
   - Minimal code snippet or command used.
   - Full stack trace / error logs.
   - Hardware and environment details (`python minival.py info`).

---

## Proposing Enhancements

MiniVal welcomes ideas for educational improvements and optimizations! Before writing extensive code for a major feature:
1. Open an issue using our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
2. Explain the problem, the proposed solution, and why it fits MiniVal's lightweight educational philosophy.
3. Once aligned with maintainers, proceed with the implementation.

---

## Documentation Improvements

Documentation is just as vital as code:
- Improvements to docstrings, comments, Indonesian/English explanations, and README guides are always welcome.
- Keep comments accessible and technically precise. Avoid overly verbose filler.

---

## Code Style & Conventions

- Follow **PEP 8** style guidelines.
- Use explicit type hints (`typing`, `from __future__ import annotations`).
- Include docstrings on all public functions, classes, and CLI subcommands.
- Keep comments bilingual or easy to follow in both Indonesian and English.
- Use `os.path` / `pathlib` for cross-platform compatibility (Windows / Linux / macOS).

---

## Review & Maintainer Expectations

- **Constructive Feedback**: Maintainers will review your PR, verify CI results, and offer feedback or suggested tweaks.
- **Fast Iteration**: If revisions are requested, push commits directly to your PR branch.
- **Preservation of Purpose**: MiniVal is built for clarity and learning. Solutions that prioritize readability and clean primitives are preferred over bloated abstractions or heavy external frameworks.
