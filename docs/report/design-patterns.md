# SafeHaven — Design Patterns Writeup

> **docs/report/design-patterns.md** · Week 2 Deliverable W3  
> ~1 page covering all 6 design patterns in the Stateful Safety Pipeline (SSP).

---

## Design Patterns in the Stateful Safety Pipeline

SafeHaven implements six Gang of Four design patterns deliberately composed to create a formally-architected safety-aware chatbot. Each pattern addresses a specific software engineering problem in the safety domain.

---

### 1. Strategy — Risk-Adaptive Response Generation

**Where:** `ConcreteStrategySelector` maps FSM states to `ResponseStrategy` implementations: `SupportiveStrategy` (CALM/CONCERNED), `DeEscalationStrategy` (ELEVATED), `CrisisStrategy` (CRISIS).

**Problem solved:** Response behaviour must vary with risk level, but the orchestrator (`ChatController`) should not contain conditional logic about *how* to respond — only *whether* to respond.

**Clinical basis:** Each strategy encodes a distinct clinical framework: Motivational Interviewing (OARS) for support, DBT Distress Tolerance (TIPP/5-4-3-2-1) for de-escalation, and QPR (Question, Persuade, Refer) for crisis. The Strategy pattern makes these frameworks swappable without modifying the controller.

---

### 2. Finite State Machine — Forward-Only Risk Ratchet

**Where:** `FSMRiskEvaluator` manages four states: `CALM → CONCERNED → ELEVATED → CRISIS`. A `_RANK` dictionary enforces the monotonic constraint via `_transition_to()`, which asserts `_RANK[new] >= _RANK[current]`.

**Problem solved:** Risk must only escalate within a session, never silently de-escalate. This mirrors clinical triage protocols where an elevated-risk flag is never quietly removed mid-session.

**Novel contribution:** Prior FSM-based chatbots (MindfulDiary, ChaCha) use FSMs to model *conversation phases*. SafeHaven applies FSM to model *risk levels* — a structurally distinct use confirmed by academic review to have no prior art.

---

### 3. Pipeline — The Stateful Safety Pipeline (SSP)

**Where:** `ChatController.handle_message()` implements an 11-step sequential pipeline: Language Detection → Emotion Detection → Memory Storage → State Construction → Risk Evaluation → Strategy Selection → Prompt Building → LLM Generation → Output Filtering → Post-Processing → Memory Storage.

**Problem solved:** Each processing stage has a single responsibility and a clearly-defined interface. No stage can bypass or shortcut another, ensuring the safety properties of earlier stages are always enforced.

**Key rule:** The LLM is invoked at step 8 only if risk ≤ MEDIUM. The pipeline enforces this by returning `None` at step 6 for HIGH risk, causing the UI to show the crisis screen without ever reaching the generator.

---

### 4. Observer — Real-Time Dashboard Updates

**Where:** `DashboardViewModel(EventDispatcher)` exposes Kivy `ObjectProperty` fields that drive the `InsightsScreen`. The `ChatController` notifies the dashboard via callback on every response, updating emotion trend bars, the risk timeline, and the conversation log.

**Problem solved:** The UI must react to controller state without the controller having any reference to the UI. The Observer pattern decouples these layers, enabling the insights dashboard to update in real time without polling.

---

### 5. Repository — Swappable Conversation Storage

**Where:** The `ConversationMemory` Protocol defines `store_message()`, `get_recent_messages()`, and `clear()`. Two concrete implementations exist: `SQLiteMemory` (production) and `InMemoryConversationMemory` (testing).

**Problem solved:** The controller should not depend on a specific storage technology. By depending only on the `ConversationMemory` Protocol, the same controller works with SQLite in production and with an in-memory store in 86 unit tests — no test ever touches the database.

---

### 6. Dependency Injection — Testable, Swappable Components

**Where:** `ChatController.__init__()` accepts all dependencies (detector, evaluator, memory, generator, output filter, language detector, strategy selector) as Protocol-typed parameters.

**Problem solved:** Every module can be replaced with a stub or mock in tests without modifying the controller. The 86-test suite injects `FakeDetector`, `FakeGenerator`, and `InMemoryConversationMemory` to test all pipeline paths — including crisis detection and FSM escalation — without an API key or database.

---

### Summary

| Pattern | Where | Benefit |
|---------|-------|---------|
| Strategy | `ConcreteStrategySelector` + 3 strategies | Clinical framework selection per risk level |
| FSM | `FSMRiskEvaluator` | Forward-only risk ratchet |
| Pipeline | `ChatController.handle_message()` | Enforced 11-step safety ordering |
| Observer | `DashboardViewModel` | Decoupled real-time UI updates |
| Repository | `ConversationMemory` (SQLite + InMemory) | Storage-agnostic controller |
| DI | `ChatController.__init__()` | Full testability without infrastructure |

These six patterns are not incidental choices — they are the architectural response to the safety gaps identified in peer-reviewed evaluations of existing chatbots (Pichowicz et al., *Scientific Reports* 2025; TherapyProbe arXiv 2602.22775).
