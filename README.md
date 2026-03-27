# SafeHaven

Safety-aware mental health chatbot built with Python and the Anthropic Claude API. Uses a Kivy UI with keyword-based risk assessment. FSM-based risk evaluation and strategy-driven response generation are architected but not yet wired in.

> **Disclaimer:** This is a CS 6221 course project — not for production or clinical use.

## Prerequisites

- **Python 3.11+**
- **Git**
- **Kivy system dependencies** — see [Kivy installation guide](https://kivy.org/doc/stable/gettingstarted/installation.html) for your platform
- (Optional) [uv](https://docs.astral.sh/uv/) for faster dependency management
- (Optional) [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) for MCP server integration

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd SafeHaven

# Create and activate a virtual environment
python -m venv .venv

# Windows (Git Bash / MSYS2)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# For local model support (Ollama)
pip install -e ".[local]"

# Configure environment variables
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY

# Run the application
python -m safehaven.main
```

## Development

```bash
# Type checking (strict mode)
python -m mypy --strict safehaven/

# Run tests
python -m pytest
```

All tests mock the LLM — no API key needed to run the test suite.

## Claude Code Setup (Optional)

The project includes a `.mcp.json` that auto-loads the **SQLite MCP server** for querying `safehaven.db` (conversation storage).

To add the **Context7 MCP server** (user-scope, provides library documentation):

```bash
claude mcp add --transport http --scope user context7 https://mcp.context7.com/mcp
```

## Project Structure

```
safehaven/
├── controller/              # ChatController — orchestrates the pipeline
├── llm/
│   ├── claude_generator.py  # ResponseGenerator impl (Anthropic Claude)
│   └── local_generator.py   # ResponseGenerator impl (Ollama, local)
├── memory/                  # SQLite-backed conversation memory
├── safety/
│   ├── emotion_detector.py  # EmotionDetector impl
│   ├── risk_evaluator.py    # RiskEvaluator impl (keyword-based, active)
│   ├── fsm_risk_evaluator.py # FSM RiskEvaluator (stub — not yet active)
│   ├── language_detector.py # LanguageDetector (stub — not yet wired)
│   └── output_filter.py     # OutputFilter impl
├── strategy/
│   ├── base.py              # ConcreteStrategySelector (stub — not yet wired)
│   ├── supportive.py        # SupportiveStrategy — CALM/CONCERNED (stub)
│   ├── de_escalation.py     # DeEscalationStrategy — ELEVATED (stub)
│   └── crisis.py            # CrisisStrategy — CRISIS (stub)
├── ui/
│   ├── app.py               # Kivy App + ScreenManager
│   ├── welcome_screen.py    # Splash/welcome screen
│   ├── chat_screen.py       # Main chat interface
│   ├── crisis_screen.py     # Crisis resources display
│   ├── insights_screen.py   # Emotional trends dashboard (placeholder)
│   └── theme.py             # Colors, emotion-to-color map
├── tests/                   # pytest test suite
├── resources/
│   ├── crisis_keywords.txt      # English crisis keywords
│   ├── crisis_keywords_ar.txt   # Arabic crisis keywords
│   ├── crisis_hotlines.json     # Country → hotline mapping
│   ├── emotion_keywords_ar.json # Arabic emotion word sets
│   └── safehaven.kv             # Unused Kivy DSL skeleton (UI is built in Python)
├── logging_config.py        # Structured logging setup
├── interfaces.py            # Protocol definitions
└── models.py                # Dataclass models
```

## Pipeline

**Current:**
```
UI (Kivy) → EmotionDetector → KeywordRiskEvaluator → ResponseGenerator → OutputFilter → UI
```

**Target (planned):**
```
UI (Kivy) → LanguageDetector → EmotionDetector → FSM RiskEvaluator → StrategySelector → ResponseGenerator → OutputFilter → UI
```

## Design Patterns

| Pattern | Where | Status | Why |
|---------|-------|--------|-----|
| **Strategy** | `StrategySelector` picks `ResponseStrategy` by FSM state | Stub — not yet wired | Swap response behavior without changing controller |
| **FSM** | `FSMRiskEvaluator` tracks escalation states | Stub — `KeywordRiskEvaluator` is active | Stateful risk assessment across conversation turns |
| **Pipeline** | `ChatController.handle_message()` | Implemented (partial — no LanguageDetector/StrategySelector yet) | Each stage transforms data for the next |
| **Observer** | UI ← Controller (callback on response) | Implemented | Decouples UI from business logic |
| **Repository** | `ConversationMemory` | Implemented | Abstracts storage (SQLite today, anything tomorrow) |
| **Dependency Injection** | Controller accepts Protocol-typed dependencies | Implemented | Easy testing with mocks, swappable implementations |
