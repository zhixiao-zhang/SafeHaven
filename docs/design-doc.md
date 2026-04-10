# SafeHaven — Design Document

> **Course Project — Safety-Aware Mental Health Chatbot**
> Version 2.1 · March 2026 — revised for 2-week sprint

---

## 1. Project Overview

**SafeHaven** is a desktop chatbot that provides empathetic conversational support while actively monitoring for signs of emotional distress or crisis. It uses an API-based LLM (Anthropic Claude) for response generation, layered behind a safety pipeline that analyzes emotion, evaluates risk, and filters output.

> **Stateful Safety Pipeline (SSP):** The full SSP is now active. Pipeline: `UI (Kivy) → LanguageDetector → EmotionDetector → FSMRiskEvaluator → StrategySelector → ResponseGenerator → OutputFilter → UI`. All components are implemented and wired.

### Scope

| In Scope | Out of Scope |
|----------|--------------|
| English + Arabic text chat | Other languages |
| Desktop GUI (Kivy) | Web or mobile deployment |
| API-based LLM (Anthropic Claude) | Fine-tuned models |
| Local LLM option (Ollama) | Cloud-hosted local models |
| FSM-based risk detection | Clinical-grade NLP |
| Strategy-driven response generation | Hardcoded response logic |
| SQLite conversation storage | Cloud database / auth |
| Crisis resource display | Actual crisis intervention |
| Emotional insights dashboard | Real-time analytics |
| 3 demo scenarios | Production deployment |

**Important disclaimer:** SafeHaven is a course project demonstrating safety-aware architecture. It is **not** a clinical tool and must not be used as a substitute for professional mental health support.

---

## 2. Architecture Overview

### Layered Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   PRESENTATION                       │
│                Kivy ScreenManager                    │
│   (welcome, chat, crisis, insights screens)          │
└──────────────────────┬──────────────────────────────┘
                       │ user text
                       ▼
┌─────────────────────────────────────────────────────┐
│                 LANGUAGE DETECTION                    │
│         LanguageDetector.detect_language(text)        │
│              → 'en' | 'ar'                           │
└──────────────────────┬──────────────────────────────┘
                       │ language code
                       ▼
┌─────────────────────────────────────────────────────┐
│                   PROCESSING                         │
│   ┌─────────────────┐   ┌────────────────────┐      │
│   │ EmotionDetector  │   │ ConversationMemory │      │
│   │ detect(text)     │   │ store / retrieve   │      │
│   └────────┬────────┘   └────────┬───────────┘      │
│            │ EmotionResult       │ recent messages   │
│            ▼                     ▼                   │
│         ┌──────────────────────────┐                 │
│         │       UserState          │                 │
│         └────────────┬─────────────┘                 │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    DECISION                          │
│           FSMRiskEvaluator (stateful)                │
│     evaluate(UserState) → RiskLevel                  │
│     FSM: CALM → CONCERNED → ELEVATED → CRISIS       │
└──────────────────────┬──────────────────────────────┘
                       │ RiskLevel + FSM state
                       ▼
┌─────────────────────────────────────────────────────┐
│               STRATEGY SELECTION                     │
│     StrategySelector.select(risk, fsm_state)         │
│        → ResponseStrategy                            │
│                                                      │
│     CALM/CONCERNED → SupportiveStrategy              │
│     ELEVATED       → DeEscalationStrategy            │
│     CRISIS         → CrisisStrategy                  │
└──────────────────────┬──────────────────────────────┘
                       │ strategy
                       ▼
              ┌────────┴────────┐
              │                 │
         LOW / MEDIUM          HIGH
              │                 │
              ▼                 ▼
┌─────────────────────┐  ┌──────────────┐
│     GENERATION      │  │ CRISIS PATH  │
│  ResponseGenerator  │  │ Lock input,  │
│  generate(context)  │  │ show crisis  │
│  (system prompt     │  │ screen with  │
│   from strategy)    │  │ hotlines     │
└─────────┬───────────┘  └──────────────┘
          │ raw response
          ▼
┌─────────────────────────────────────────────────────┐
│                   VALIDATION                         │
│   OutputFilter.validate(response, risk) → str        │
│   + Strategy.post_process(response) → str            │
│   (strip harmful content, enforce tone)              │
└──────────────────────┬──────────────────────────────┘
                       │ safe response
                       ▼
┌─────────────────────────────────────────────────────┐
│                   PRESENTATION                       │
│   Display response in chat (emotion-colored bubble)  │
└─────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Key Rule | Status |
|-------|---------|----------|--------|
| **Presentation** | Renders UI, captures input | Never calls LLM directly | ✅ Implemented — Kivy, 4 screens; FSM color bar + emotion-colored bubbles |
| **Language Detection** | Identifies input language | Runs before emotion detection | ✅ Implemented — `SimpleLanguageDetector` (Unicode Arabic ratio) |
| **Processing** | Extracts emotion, stores messages | Stateless detection, stateful storage | ✅ Implemented |
| **Decision** | Evaluates risk from UserState | Single source of risk truth | ✅ Implemented — `FSMRiskEvaluator` active; forward-only ratchet constraint |
| **Strategy Selection** | Picks response strategy by FSM state | Decouples strategy from controller logic | ✅ Implemented — `ConcreteStrategySelector` + 3 strategies wired in controller |
| **Generation** | Calls LLM API with strategy-built prompt | Only called if risk ≤ MEDIUM | ✅ Implemented (Claude + Ollama) |
| **Validation** | Filters LLM output + strategy post-processing | Last gate before display | ✅ Implemented |

---

## 3. Data Models

All models live in `safehaven/models.py`.

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class FSMState(Enum):
    CALM = "calm"
    CONCERNED = "concerned"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class EmotionLabel(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"


class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class EmotionResult:
    label: EmotionLabel
    confidence: float  # 0.0 – 1.0


@dataclass
class Message:
    role: str              # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: EmotionLabel | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    language: str = "en"


@dataclass
class UserState:
    current_emotion: EmotionResult
    risk_level: RiskLevel
    message_count: int
    escalation_history: list[RiskLevel] = field(default_factory=list)
    language: str = "en"
    fsm_state: str = "calm"


@dataclass
class ConversationContext:
    recent_messages: list[Message]
    user_state: UserState
    system_prompt: str = ""
    strategy_name: str = ""

    def to_llm_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.recent_messages]
```

---

## 4. Module Interfaces

Each module is defined as a Python Protocol so implementations can be swapped (real vs. stub). All interfaces live in `safehaven/interfaces.py`.

```python
from typing import Protocol
from safehaven.models import (
    EmotionResult, RiskLevel, UserState,
    Message, ConversationContext,
)


class EmotionDetector(Protocol):
    def detect(self, text: str) -> EmotionResult:
        """Analyze text and return the dominant emotion with confidence."""
        ...


class RiskEvaluator(Protocol):
    def evaluate(self, state: UserState) -> RiskLevel:
        """Determine risk level from current user state."""
        ...


class ConversationMemory(Protocol):
    def store_message(self, message: Message) -> None:
        """Persist a message to storage."""
        ...

    def get_recent_messages(self, limit: int = 10) -> list[Message]:
        """Retrieve the N most recent messages, oldest first."""
        ...

    def clear(self) -> None:
        """Clear all stored messages (new session)."""
        ...


class ResponseGenerator(Protocol):
    def generate(self, context: ConversationContext) -> str:
        """Call the LLM API and return the raw response text."""
        ...


class OutputFilter(Protocol):
    def validate(self, response: str, risk: RiskLevel) -> str:
        """Sanitize LLM output based on current risk level."""
        ...


class LanguageDetector(Protocol):
    def detect_language(self, text: str) -> str:
        """Return ISO 639-1 language code ('en', 'ar')."""
        ...


class ResponseStrategy(Protocol):
    def build_system_prompt(self, context: ConversationContext) -> str:
        """Return a system prompt tailored to current risk/emotion state."""
        ...

    def post_process(self, response: str) -> str:
        """Optional post-processing of LLM output."""
        ...


class StrategySelector(Protocol):
    def select(self, risk: RiskLevel, fsm_state: str) -> ResponseStrategy:
        """Pick appropriate strategy based on FSM state."""
        ...
```

### ChatController — Orchestrator (SSP)

The controller implements the full 11-step Stateful Safety Pipeline:

```python
# handle_message() pipeline steps:
# 1.  Detect language (SimpleLanguageDetector)
# 2.  Detect emotion (KeywordEmotionDetector)
# 3.  Store user message (with language + emotion)
# 4.  Build UserState (escalation_history from memory)
# 5.  Evaluate risk (FSMRiskEvaluator — stateful, forward-only)
# 6.  Crisis path: return None → UI shows crisis screen
# 7.  Select strategy (ConcreteStrategySelector by fsm_state)
# 8.  Build system prompt (strategy.build_system_prompt())
# 9.  Generate response (LLM with strategy system prompt)
# 10. Filter output (SafeOutputFilter)
# 11. Post-process (strategy.post_process()) → store + return
```

Controller exposes:
- `controller.fsm_state` → current FSM state string (for UI color bar)
- `controller.last_emotion` → last detected emotion (for bubble color)
- `controller.clear()` → resets memory + FSM for new session

---

## 5. Folder Structure

```
safehaven/
├── main.py                      # Entry point — launches UI
├── models.py                    # Data models (Section 3)
├── interfaces.py                # Protocol definitions (Section 4)
├── logging_config.py            # Structured logging setup
├── ui/
│   ├── __init__.py
│   ├── app.py                   # Kivy App + ScreenManager
│   ├── welcome_screen.py        # Splash/welcome screen
│   ├── chat_screen.py           # Main chat interface
│   ├── crisis_screen.py         # Crisis resource display
│   ├── insights_screen.py       # Emotional trends dashboard (DashboardViewModel + Observer)
│   └── theme.py                 # Colors, emotion-to-color map
├── memory/
│   ├── __init__.py
│   ├── sqlite_memory.py         # ConversationMemory impl (SQLite — production)
│   ├── in_memory.py             # ConversationMemory impl (in-memory — testing)
│   └── schema.sql               # CREATE TABLE statements
├── safety/
│   ├── __init__.py
│   ├── emotion_detector.py      # EmotionDetector impl
│   ├── risk_evaluator.py        # RiskEvaluator impl (keyword-based)
│   ├── fsm_risk_evaluator.py    # FSM RiskEvaluator impl (stateful)
│   ├── language_detector.py     # LanguageDetector impl
│   └── output_filter.py         # OutputFilter impl
├── llm/
│   ├── __init__.py
│   ├── claude_generator.py      # ResponseGenerator impl (Anthropic Claude)
│   └── local_generator.py       # ResponseGenerator impl (Ollama, local)
├── strategy/
│   ├── __init__.py
│   ├── base.py                  # ConcreteStrategySelector
│   ├── supportive.py            # SupportiveStrategy (CALM/CONCERNED)
│   ├── de_escalation.py         # DeEscalationStrategy (ELEVATED)
│   └── crisis.py                # CrisisStrategy (CRISIS)
├── controller/
│   ├── __init__.py
│   └── chat_controller.py       # ChatController (orchestrator)
├── tests/
│   ├── __init__.py
│   ├── test_emotion.py
│   ├── test_risk.py
│   ├── test_filter.py
│   ├── test_memory.py
│   ├── test_controller.py       # Integration test with stubs
│   ├── test_fsm.py              # FSM transition tests
│   ├── test_strategy.py         # Strategy pattern tests
│   ├── test_language.py         # Language detection tests
│   └── test_local_generator.py  # Local model tests
├── resources/
│   ├── crisis_hotlines.json     # Country → hotline mapping
│   ├── crisis_keywords.txt      # English crisis keywords
│   ├── crisis_keywords_ar.txt   # Arabic crisis keywords
│   ├── emotion_keywords_ar.json # Arabic emotion word sets
│   └── safehaven.kv             # Unused Kivy DSL skeleton (UI is built in Python)
CLAUDE.md                            # Shared LLM context for team (repo root)
```

---

## 6. Sprint Plan — 2-Week Task Board

> **Context:** Weeks 1–4 of the original plan are complete. The codebase has working infrastructure (models, interfaces, controller, UI, memory, emotion detection, keyword risk, output filter, Claude/Ollama generators). The remaining work is implementing the three stubbed modules, wiring them into the controller, and preparing the demo and report.

> **Principle:** Each task has a binary "done" check. Aim for ~3 hours per task. The critical path runs through Backend A — unblock it first.

---

### Team Assignments

| Group | Person(s) | Owns |
|-------|-----------|------|
| **Backend A** | 1 person | `FSMRiskEvaluator`, `LanguageDetector` |
| **Backend B** | 1 person | `SupportiveStrategy`, `DeEscalationStrategy`, `CrisisStrategy`, `ConcreteStrategySelector` |
| **Backend C** | 1 person | Wire `ChatController` (inject FSM + strategies + language), integration tests |
| **Frontend** | Quoc | `InsightsScreen`, emotion-colored bubbles, Arabic layout hint, UI polish |
| **Data & Tests** | Quoc & Patrick | Keyword lists, unit tests for new modules, demo script, adversarial filter tests |
| **Docs** | Quoc | UML diagrams, report assembly, presentation slides |

> All three backend members work in parallel from Day 1. A and B agree upfront on the `fsm_state` string contract (`"calm"`, `"concerned"`, `"elevated"`, `"crisis"`). C starts by scaffolding integration tests against mocks and wires the controller once A and B finish their stubs. Frontend, Data & Tests, and Docs are fully independent from Day 1.

---

### Dependency / Critical Path

```
Backend A: FSMRiskEvaluator + LanguageDetector  ─┐
Backend B: Strategies + StrategySelector         ─┤ (all parallel, Week 1)
Backend C: Test scaffolding (mocked)             ─┘
                ↓ (A + B done)
Backend C: Wire ChatController
                ↓
Backend C: Integration tests (real impls)
```

Frontend → InsightsScreen depends on `ConversationMemory` (already done) — no backend dependency.

---

### ~~Week 1 — Implement Stubs + Wire Controller~~ ✅ COMPLETE

#### Backend A ✅

| # | Task | Status |
|---|------|--------|
| A1 | Implement `FSMRiskEvaluator.evaluate()` — CALM→CONCERNED→ELEVATED→CRISIS transitions | ✅ Done — 15 tests passing |
| A2 | Implement `SimpleLanguageDetector.detect_language()` — Unicode Arabic script check | ✅ Done — 9 tests passing |

#### Backend B ✅

| # | Task | Status |
|---|------|--------|
| B1 | Implement `SupportiveStrategy.build_system_prompt()` — MI/OARS prompt for CALM/CONCERNED | ✅ Done |
| B2 | Implement `DeEscalationStrategy.build_system_prompt()` — DBT/TIPP prompt for ELEVATED | ✅ Done |
| B3 | Implement `CrisisStrategy.build_system_prompt()` — QPR prompt for CRISIS state | ✅ Done |
| B4 | Implement `ConcreteStrategySelector.select()` — maps `fsm_state` to strategy | ✅ Done — 4 selector tests passing |

#### Backend C ✅

| # | Task | Status |
|---|------|--------|
| C1 | Integration test scaffolding in `test_controller.py` | ✅ Done |
| C2 | Wire `ChatController` with `LanguageDetector` + `FSMRiskEvaluator` + `StrategySelector` | ✅ Done — 11-step SSP active |

#### Frontend ✅

| # | Task | Status |
|---|------|--------|
| F1 | Build `InsightsScreen` — `DashboardViewModel` + emotion bars + risk timeline + log panel | ✅ Done |
| F2 | Emotion-colored message bubbles + FSM state indicator bar with animation | ✅ Done |
| F3 | "Thinking…" indicator while LLM call is in-flight | ✅ Done |

#### Data & Tests ✅

| # | Task | Status |
|---|------|--------|
| D1 | Expand `crisis_keywords.txt` to 50+ entries | ✅ Done — 58 entries |
| D2 | Expand `crisis_keywords_ar.txt` to 20+ entries | ✅ Done — 32 entries |
| D3 | `test_fsm.py` — 15 tests covering transitions, terminal state, skip-states, clear | ✅ Done |
| D4 | `test_strategy.py` — 17 tests covering all 3 strategies + selector | ✅ Done |
| D5 | `test_language.py` — 9 tests covering English, Arabic, mixed, edge cases | ✅ Done |

---

### Week 2 — Integration, Testing, Demo, Documentation

#### Backend C (+ A and B as needed)

| # | Task | Status |
|---|------|--------|
| I1 | End-to-end integration test: Scenario 1 (normal conversation, 3 turns, LOW risk) | ✅ Done — `test_integration.py` |
| I2 | End-to-end integration test: Scenario 2 (distressed user, MEDIUM risk, CONCERNED→ELEVATED) | ✅ Done — `test_integration.py` |
| I3 | End-to-end integration test: Scenario 3 (crisis detection, HIGH risk) | ✅ Done — `test_integration.py` |
| I4 | Error handling: API timeout, empty input, SQLite lock | ✅ Done — `test_integration.py` |

#### Frontend

| # | Task | Status |
|---|------|--------|
| F4 | ~~"thinking…" indicator~~ ✅ done · Timestamps on messages | Timestamps visible on each bubble |
| F5 | Manual smoke test: run all 3 demo scenarios live with real API key | Screenshots captured for `docs/demo/` |

#### Data & Tests

| # | Task | Status |
|---|------|--------|
| D6 | Write demo script — exact user inputs and expected outputs for all 3 scenarios | ✅ Done — `docs/demo/demo-script.md` |
| D7 | Adversarial output filter tests — 5+ LLM responses containing harmful patterns | ✅ Done — `test_adversarial_filter.py`, 17 tests passing |
| D8 | `mypy --strict safehaven/` passes with zero errors after all new code | ✅ Done — 0 errors, 43 source files |

#### Docs

| # | Task | Status |
|---|------|--------|
| W1 | Final UML class diagram (PlantUML or draw.io) — reflects actual wired code | PNG/SVG in `docs/`; all 6 design patterns annotated |
| W2 | Sequence diagram for `handle_message()` through the full target pipeline | PNG/SVG in `docs/` |
| W3 | Design patterns writeup — 1 page covering Strategy, FSM, Pipeline, Observer, Repository, DI | ✅ Done — `docs/report/design-patterns.md` |
| W4 | Individual report sections — each person writes ~1 page on their contribution | Sections in `docs/report/` |
| W5 | Presentation slides (5–8 slides): architecture, patterns, demo, learnings | Dry-run rehearsal done; everyone knows their part |
| W6 | Final report assembly + proofread | Single PDF ready for submission |

---

## 7. Design Patterns Used

| Pattern | Where | Clinical Basis | Status |
|---------|-------|----------------|--------|
| **Strategy** | `ConcreteStrategySelector` picks `ResponseStrategy` by FSM state; strategies: `SupportiveStrategy`, `DeEscalationStrategy`, `CrisisStrategy` | MI (OARS) + DBT (TIPP/5-4-3-2-1) + QPR — each strategy grounded in clinical framework | ✅ Implemented + wired in controller |
| **Finite State Machine** | `FSMRiskEvaluator` manages states: CALM → CONCERNED → ELEVATED → CRISIS; `_RANK` dict enforces monotonic (forward-only) constraint | Ratchet constraint — mirrors clinical escalation-to-crisis protocols | ✅ Implemented — active evaluator |
| **Pipeline** | `ChatController.handle_message()` — 11-step SSP: Language → Emotion → Memory → State → Risk → Strategy → Prompt → LLM → Filter → PostProcess → Store | Layered safety analogous to clinical triage protocols | ✅ Full SSP implemented |
| **Observer** | `DashboardViewModel(EventDispatcher)` properties drive `InsightsScreen` UI updates; UI ← Controller (callback on response) | — | ✅ Implemented (two observers: chat response + dashboard) |
| **Repository** | `ConversationMemory` protocol — two backends: `SQLiteMemory` (production) + `InMemoryConversationMemory` (testing) | — | ✅ Implemented (two concrete backends) |
| **Dependency Injection** | Controller accepts Protocol-typed dependencies; all modules swappable without changing orchestrator | — | ✅ Implemented — 86 tests use injected fakes |

### FSM State Diagram

```
          negative emotion         sustained negative        crisis keyword
  ┌──────┐  (confidence > 0.6)  ┌───────────┐  (3+ turns) ┌──────────┐         ┌────────┐
  │ CALM │ ──────────────────→  │ CONCERNED │ ──────────→  │ ELEVATED │ ──────→ │ CRISIS │
  └──────┘                      └───────────┘              └──────────┘         └────────┘
     │                               │                          │                    │
     └── neutral emotion ←───────────┘ (no de-escalation)       │                    │
         stays in CALM               stays or advances          stays or advances    terminal
```

**Transition rules:**
- CALM → CONCERNED: Negative emotion detected with confidence > 0.6
- CONCERNED → ELEVATED: 3+ consecutive negative emotion turns
- ELEVATED → CRISIS: Crisis keyword detected or rapid escalation
- CRISIS: Terminal state for the session (reset only via `clear()`)
- No backward transitions within a session

---

## 8. Demo Scenarios

### Scenario 1 — Normal Conversation (LOW risk, CALM state)

```
User:  "Hi! I had a pretty good day today."
       → LanguageDetector: 'en'
       → EmotionDetector: HAPPY (0.85)
       → FSMRiskEvaluator: CALM → stays CALM
       → StrategySelector: SupportiveStrategy
       → RiskLevel: LOW
       → LLM generates friendly response (supportive prompt)
       → OutputFilter: passes through unchanged

Bot:   "That's great to hear! What made your day good?"
```

### Scenario 2 — Distressed User (MEDIUM risk, CONCERNED → ELEVATED)

```
User:  "I've been feeling really overwhelmed lately. Nothing seems to work out."
       → LanguageDetector: 'en'
       → EmotionDetector: SAD (0.78)
       → FSMRiskEvaluator: CALM → CONCERNED (negative emotion, high confidence)
       → StrategySelector: SupportiveStrategy
       → RiskLevel: MEDIUM
       → LLM generates response with empathetic system prompt
       → OutputFilter: prepends empathetic framing

Bot:   "It sounds like you're going through a really tough time, and that's
        completely valid. Would you like to talk about what's been weighing
        on you most?"
```

### Scenario 3 — Crisis Detection (HIGH risk, CRISIS state)

```
User:  "I don't want to be here anymore. I want to end it all."
       → LanguageDetector: 'en'
       → EmotionDetector: FEARFUL (0.92)
       → FSMRiskEvaluator: → CRISIS (crisis keyword match)
       → StrategySelector: CrisisStrategy
       → RiskLevel: HIGH
       → ChatController returns None
       → UI shows crisis screen

┌─────────────────────────────────────────┐
│        ⚠ We Care About You              │
│                                         │
│  It sounds like you may be in crisis.   │
│  Please reach out to a professional:    │
│                                         │
│  • 988 Suicide & Crisis Lifeline        │
│    Call or text: 988                     │
│  • Crisis Text Line                     │
│    Text HOME to 741741                  │
│  • International Association for        │
│    Suicide Prevention:                  │
│    https://www.iasp.info/resources/     │
│                                         │
│  [ I understand — continue chatting ]   │
└─────────────────────────────────────────┘
```

---

## 9. Interface Compatibility Checklist

| Producer | Output | Consumer | Input | Match? |
|----------|--------|----------|-------|--------|
| `LanguageDetector.detect_language()` | `str` ('en'/'ar') | `UserState.language` | `str` | ✓ |
| `EmotionDetector.detect()` | `EmotionResult` | `UserState` constructor | `EmotionResult` | ✓ |
| `UserState` | `UserState` | `RiskEvaluator.evaluate()` | `UserState` | ✓ |
| `RiskEvaluator.evaluate()` | `RiskLevel` | `StrategySelector.select()` | `RiskLevel` | ✓ |
| `FSMRiskEvaluator.state` | `str` | `StrategySelector.select()` | `str` (fsm_state) | ✓ |
| `StrategySelector.select()` | `ResponseStrategy` | `ResponseStrategy.build_system_prompt()` | self | ✓ |
| `ConversationMemory.get_recent_messages()` | `list[Message]` | `ConversationContext` | `list[Message]` | ✓ |
| `ConversationContext.to_llm_messages()` | `list[dict]` | Claude API | `list[dict]` | ✓ |
| `ResponseStrategy.build_system_prompt()` | `str` | `ConversationContext.system_prompt` | `str` | ✓ |
| `ResponseGenerator.generate()` | `str` | `OutputFilter.validate()` | `str` | ✓ |
| `ResponseStrategy.post_process()` | `str` | Final response | `str` | ✓ |
| `RiskEvaluator.evaluate()` | `RiskLevel` | `OutputFilter.validate()` | `RiskLevel` | ✓ |
| `ChatController.handle_message()` | `str \| None` | UI callback | `str \| None` | ✓ |

---

*End of design document.*
