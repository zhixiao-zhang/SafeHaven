# SafeHaven — CLAUDE.md

## Commit Style

Never add `Co-Authored-By` or any AI attribution lines to commit messages in this repo.

## Project

SafeHaven is a safety-aware mental health chatbot (Python, Anthropic Claude API). Course project — not for production use. Supports English and Arabic.

## Architecture

**Current pipeline:** `UI (Kivy) → EmotionDetector → KeywordRiskEvaluator → ResponseGenerator → OutputFilter → UI`

**Target pipeline (planned):** `UI (Kivy) → LanguageDetector → EmotionDetector → FSM RiskEvaluator → StrategySelector → ResponseGenerator → OutputFilter → UI`

> LanguageDetector, FSMRiskEvaluator, and StrategySelector are stubbed but not yet wired into the controller.

`ChatController.handle_message` returns `None` on HIGH risk — UI must show crisis screen, not a text response.

## FSM States (planned — not yet active)

`CALM → CONCERNED → ELEVATED → CRISIS`

- **CALM**: Normal conversation, low risk signals
- **CONCERNED**: Mild negative emotion detected, monitoring
- **ELEVATED**: Sustained negative emotion or escalation pattern
- **CRISIS**: Crisis keywords detected or rapid escalation

Transitions are forward-only within a session (no backwards movement). Reset on `clear()`.

> FSM states are defined in `models.py` and `FSMRiskEvaluator` exists but raises `NotImplementedError`. The active evaluator is `KeywordRiskEvaluator` (stateless, keyword-based).

## Strategy Pattern (planned — not yet wired)

Strategy is selected by FSM state via `StrategySelector`:

| FSM State | Strategy | Behavior |
|-----------|----------|----------|
| CALM / CONCERNED | `SupportiveStrategy` | Warm, empathetic prompting |
| ELEVATED | `DeEscalationStrategy` | Grounding, safety-aware prompting |
| CRISIS | `CrisisStrategy` | Minimal LLM, directs to resources |

> Strategies and `ConcreteStrategySelector` are defined but stubbed (`NotImplementedError`). `ChatController` does not yet call `StrategySelector` — no strategy-driven prompting occurs.

## Multilingual

- `LanguageDetector` is stubbed (`NotImplementedError`) — not yet integrated into the pipeline
- Per-language keyword files: `crisis_keywords.txt` / `crisis_keywords_ar.txt`
- Per-language emotion words: `emotion_keywords_ar.json`

## Conventions

- Python 3.11+, `mypy --strict` must pass
- Modules implement **Protocols** from `interfaces.py` — don't subclass, just match the signature
- All data models live in `models.py` as dataclasses, not dicts
- Tests: `pytest`, always mock the LLM
- Crisis keywords in `resources/crisis_keywords.txt` (English) and `resources/crisis_keywords_ar.txt` (Arabic), one per line
- Emotion keywords for Arabic in `resources/emotion_keywords_ar.json`

## MCP Servers

- **SQLite** (project, `.mcp.json`): queries `safehaven.db` — the conversation storage database

## Gotchas

- Never call the LLM directly from UI code — always go through `ChatController`
- All crisis-related logic goes through `RiskEvaluator`, not ad-hoc keyword checks
- Strategy selection will go through `StrategySelector`, not hardcoded if/else (not yet wired)
- FSM will be stateful — lives for the session, reset on `clear()` (not yet active; `KeywordRiskEvaluator` is used)
