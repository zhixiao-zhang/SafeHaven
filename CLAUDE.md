# SafeHaven — CLAUDE.md

## Project

SafeHaven is a safety-aware mental health chatbot (Python, Tkinter, Anthropic Claude API). Course project — not for production use.

## Architecture

Pipeline: **UI → EmotionDetector → RiskEvaluator → ResponseGenerator → OutputFilter → UI**

`ChatController.handle_message` returns `None` on HIGH risk — UI must show crisis modal, not a text response.

## Conventions

- Python 3.11+, `mypy --strict` must pass
- Modules implement **Protocols** from `interfaces.py` — don't subclass, just match the signature
- All data models live in `models.py` as dataclasses, not dicts
- Tests: `pytest`, always mock the LLM
- Crisis keywords in `resources/crisis_keywords.txt` (one per line)

## Gotchas

- Never call the LLM directly from UI code — always go through `ChatController`
- All crisis-related logic goes through `RiskEvaluator`, not ad-hoc keyword checks
