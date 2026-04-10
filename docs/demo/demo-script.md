# SafeHaven — Demo Script

> **docs/demo-script.md** · Sprint Week 2 Deliverable D6  
> Use this script for the live demo presentation. Each scenario has exact
> user inputs and expected system outputs.

---

## Pre-Demo Setup Checklist

- [ ] `.env` contains a valid `ANTHROPIC_API_KEY`
- [ ] `python -m safehaven.main` launches without errors
- [ ] App window is visible and on the Welcome screen
- [ ] Screen recording or projector is active

---

## Scenario 1 — Normal Conversation (LOW risk · CALM state)

**Goal:** Show normal supportive conversation. FSM stays CALM, LLM responds.

| Turn | Who | Input |
|------|-----|-------|
| 1 | User | `Hi! I had a pretty good day today.` |
| 2 | Bot | *Friendly response asking about their day* |
| 3 | User | `Work went well and I had lunch with a friend.` |
| 4 | Bot | *Warm, conversational reply* |
| 5 | User | `I'm just relaxing now. Thanks for chatting!` |
| 6 | Bot | *Supportive closing* |

**Expected pipeline behaviour:**

```
Input: "Hi! I had a pretty good day today."
→ LanguageDetector:    'en'
→ EmotionDetector:     HAPPY (0.85)
→ FSMRiskEvaluator:    CALM → stays CALM
→ StrategySelector:    SupportiveStrategy (MI/OARS)
→ RiskLevel:           LOW
→ LLM generates:       friendly response
→ OutputFilter:        passes through unchanged
```

**FSM state bar:** 🟢 Green  
**Emotion bubble:** 🟡 Yellow/warm  
**Key talking point:** "Every message passes through all 7 pipeline stages. No component ever calls the LLM directly — only the controller does, and only after safety evaluation."

---

## Scenario 2 — Distressed User (MEDIUM risk · CONCERNED → ELEVATED)

**Goal:** Show FSM escalation and strategy switching when user expresses distress.

| Turn | Who | Input |
|------|-----|-------|
| 1 | User | `I've been feeling really overwhelmed lately. Nothing seems to work out.` |
| 2 | Bot | *Empathetic response; FSM now CONCERNED* |
| 3 | User | `I just feel like I'm failing at everything. It's exhausting.` |
| 4 | Bot | *Deeper validation; counter=2* |
| 5 | User | `I don't know how much longer I can keep going like this.` |
| 6 | Bot | *Grounding language (DBT/TIPP); FSM now ELEVATED; 988 appended* |

**Expected pipeline behaviour (turn 1):**

```
Input: "I've been feeling really overwhelmed lately."
→ LanguageDetector:    'en'
→ EmotionDetector:     SAD (0.78)
→ FSMRiskEvaluator:    CALM → CONCERNED (negative emotion, high confidence)
→ StrategySelector:    SupportiveStrategy (CONCERNED → Supportive, not DeEscalation)
→ RiskLevel:           MEDIUM
→ LLM generates:       empathetic response
→ OutputFilter:        prepends empathetic framing at MEDIUM
```

**Expected pipeline behaviour (turn 3, after 2 prior negative turns):**

```
→ FSMRiskEvaluator:    CONCERNED → ELEVATED (3 consecutive negatives)
→ StrategySelector:    DeEscalationStrategy (DBT/TIPP)
→ DeEscalationStrategy.post_process(): appends 988 crisis line
```

**FSM state bar:** 🟠 Orange  
**Emotion bubble:** 🔵 Blue (SAD)  
**Key talking point:** "The FSM ratchet — risk can only escalate, never retreat. Three consecutive negative turns trigger the DeEscalation strategy automatically."

---

## Scenario 3 — Crisis Detection (HIGH risk · CRISIS state)

**Goal:** Show crisis path — LLM is bypassed, crisis screen is shown immediately.

| Turn | Who | Input |
|------|-----|-------|
| 1 | User | `I don't want to be here anymore. I want to end it all.` |
| 2 | System | *→ Crisis screen appears; chat input locked* |

**Expected pipeline behaviour:**

```
Input: "I don't want to be here anymore. I want to end it all."
→ LanguageDetector:    'en'
→ EmotionDetector:     FEARFUL (0.92) — crisis keyword match
→ FSMRiskEvaluator:    → CRISIS (FEARFUL ≥ 0.9, forward-only skip)
→ StrategySelector:    CrisisStrategy
→ RiskLevel:           HIGH
→ ChatController:      returns None (LLM never called)
→ UI:                  navigates to CrisisScreen
```

**Crisis screen shows:**
```
⚠ We Care About You

It sounds like you may be in crisis.
Please reach out to a professional:

• 988 Suicide & Crisis Lifeline
  Call or text: 988

• Crisis Text Line
  Text HOME to 741741

• International Association for Suicide Prevention:
  https://www.iasp.info/resources/

[ I understand — continue chatting ]
```

**FSM state bar:** 🔴 Red  
**Key talking point:** "The LLM is never invoked during crisis. This is a deliberate architectural decision — TherapyProbe (arXiv Feb 2026) found 'Crisis Escalation Failure' as a top safety anti-pattern. Our forward-only ratchet directly addresses that."

---

## Adversarial Demo (optional, if time permits)

Show the output filter in action:

Type in the **API Response Simulator** (dev mode only):
```
Here is some advice. Take 500mg of aspirin. I understand you're struggling.
```

**Expected:** The `500mg` line is stripped. Safe content passes through.

---

## Q&A Preparation

| Question | Answer |
|----------|--------|
| "Why not just use Wysa?" | Wysa's safety logic is a black box. SafeHaven's entire pipeline is unit-tested at every stage; any component can be swapped via DI without touching the orchestrator. |
| "How is this different from MindfulDiary?" | MindfulDiary's FSM models *conversation phases*. SafeHaven's FSM models *risk level* — it tracks escalation across turns and adapts the entire response strategy. |
| "Why GoF patterns?" | This is a software engineering course. The contribution is demonstrating that patterns taught in this class solve a real safety problem in a domain where existing tools are ad hoc and systematically insufficient. |
| "What's the ratchet constraint?" | Risk can only escalate within a session, never de-escalate. Mirrors clinical protocol: once a patient is flagged as elevated risk, you don't quietly un-flag them mid-session. Reset only happens via `clear()`. |

---

*End of demo script.*
