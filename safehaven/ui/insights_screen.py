"""Insights dashboard screen — emotional trends and session risk history.

Implements the Observer pattern via Kivy's EventDispatcher/Property system:
- DashboardViewModel (EventDispatcher) is the Subject
- UI widgets bind to its properties and update automatically (Observers)
- Memory (Repository) is the data source — fetched on demand via set_memory()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kivy.animation import Animation
from kivy.event import EventDispatcher
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.properties import DictProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget

from safehaven.models import EmotionLabel, Message, RiskLevel
from safehaven.ui.theme import (
    DASHBOARD_BG,
    DASHBOARD_CARD,
    EMOTION_COLORS,
    PRIMARY_COLOR,
    RISK_COLORS,
)

if TYPE_CHECKING:
    from safehaven.interfaces import ConversationMemory


def _hex_to_rgba(hex_color: str) -> list[float]:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]


class DashboardViewModel(EventDispatcher):
    """Observable view-model for the insights dashboard.

    Implements the Observer pattern: UI widgets bind to these Kivy Properties
    and update automatically when refresh() is called.
    """

    message_count = NumericProperty(0)
    emotion_counts = DictProperty({})
    current_risk = StringProperty("calm")
    risk_history = ListProperty([])

    def refresh(self, memory: ConversationMemory) -> None:
        """Pull latest data from the Repository and update all properties."""
        messages = memory.get_recent_messages(limit=100)
        self.message_count = len(messages)

        counts: dict[str, int] = {}
        history: list[str] = []
        for m in messages:
            if m.emotion is not None:
                key = m.emotion.value
                counts[key] = counts.get(key, 0) + 1
            if m.role == "user":
                history.append(m.risk_level.name.lower())

        self.emotion_counts = counts
        self.risk_history = history
        if history:
            self.current_risk = history[-1]


class _StatCard(BoxLayout):
    """A small card widget showing a label + value."""

    def __init__(self, title: str, value: str, color: str = DASHBOARD_CARD, **kwargs: object) -> None:
        super().__init__(orientation="vertical", padding=10, spacing=4, **kwargs)
        self._bg_color = color
        with self.canvas.before:
            Color(*_hex_to_rgba(color))
            self._rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[8])
        self.bind(size=self._upd, pos=self._upd)

        self._value_lbl = Label(
            text=value,
            font_size="20sp",
            bold=True,
            color=[1, 1, 1, 1],
            size_hint_y=None,
            height=30,
        )
        self._title_lbl = Label(
            text=title,
            font_size="11sp",
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=20,
        )
        self.add_widget(self._value_lbl)
        self.add_widget(self._title_lbl)

    def set_value(self, value: str) -> None:
        self._value_lbl.text = value

    def set_color(self, color: str) -> None:
        with self.canvas.before:
            Color(*_hex_to_rgba(color))
            RoundedRectangle(size=self.size, pos=self.pos, radius=[8])

    def _upd(self, *_: object) -> None:
        self._rect.size = self.size
        self._rect.pos = self.pos


class InsightsScreen(Screen):
    """Emotion and risk insights dashboard.

    Demonstrates Observer pattern: DashboardViewModel properties drive
    all UI updates without the screen needing to know about the Repository.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._memory: ConversationMemory | None = None
        self._vm = DashboardViewModel()
        self._log_lines: list[str] = []

        with self.canvas.before:
            Color(*_hex_to_rgba(DASHBOARD_BG))
            self._bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._upd_bg, pos=self._upd_bg)

        root = BoxLayout(orientation="vertical", padding=16, spacing=12)

        # ── Header ──────────────────────────────────────────────
        header = BoxLayout(size_hint_y=None, height=44, spacing=8)
        header.add_widget(Label(
            text="Emotional Insights",
            font_size="20sp",
            bold=True,
            color=[1, 1, 1, 1],
        ))
        back_btn = Button(
            text="Back to Chat",
            size_hint_x=None,
            width=130,
            font_size="13sp",
            background_color=_hex_to_rgba(PRIMARY_COLOR),
            color=[1, 1, 1, 1],
        )
        back_btn.bind(on_release=self._go_back)
        header.add_widget(back_btn)
        root.add_widget(header)

        # ── Stat cards ──────────────────────────────────────────
        cards_row = BoxLayout(size_hint_y=None, height=80, spacing=8)
        self._card_messages = _StatCard("Messages", "0", DASHBOARD_CARD, size_hint_x=1)
        self._card_risk = _StatCard("Current Risk", "calm", RISK_COLORS["calm"], size_hint_x=1)
        self._card_turns = _StatCard("User Turns", "0", DASHBOARD_CARD, size_hint_x=1)
        cards_row.add_widget(self._card_messages)
        cards_row.add_widget(self._card_risk)
        cards_row.add_widget(self._card_turns)
        root.add_widget(cards_row)

        # ── Emotion distribution bars ────────────────────────────
        root.add_widget(Label(
            text="Emotion Distribution",
            font_size="13sp",
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=22,
            halign="left",
        ))
        self._emotion_bars: dict[str, Widget] = {}
        self._emotion_bar_container = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            spacing=4,
        )
        self._emotion_bar_container.bind(
            minimum_height=self._emotion_bar_container.setter("height")
        )
        for emotion in EmotionLabel:
            row = BoxLayout(size_hint_y=None, height=22, spacing=6)
            row.add_widget(Label(
                text=emotion.value.capitalize(),
                font_size="12sp",
                color=[0.8, 0.8, 0.8, 1],
                size_hint_x=None,
                width=70,
            ))
            bar_bg = BoxLayout(size_hint=(1, None), height=14)
            bar = Widget(size_hint=(0, None), height=14)
            with bar.canvas:
                Color(*_hex_to_rgba(EMOTION_COLORS.get(emotion, "#888888")))
                setattr(bar, "_rect", Rectangle(size=bar.size, pos=bar.pos))
            bar.bind(size=lambda inst, v: setattr(getattr(inst, "_rect"), "size", v))
            bar.bind(pos=lambda inst, v: setattr(getattr(inst, "_rect"), "pos", v))
            bar_bg.add_widget(bar)
            self._emotion_bars[emotion.value] = bar
            self._count_lbl = Label(
                text="0",
                font_size="11sp",
                color=[0.6, 0.6, 0.6, 1],
                size_hint_x=None,
                width=24,
            )
            row.add_widget(bar_bg)
            row.add_widget(self._count_lbl)
            self._emotion_bar_container.add_widget(row)
        root.add_widget(self._emotion_bar_container)

        # ── Risk timeline ────────────────────────────────────────
        root.add_widget(Label(
            text="Risk Timeline (per user turn)",
            font_size="13sp",
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=22,
            halign="left",
        ))
        timeline_scroll = ScrollView(size_hint=(1, None), height=28, do_scroll_y=False)
        self._timeline_row = BoxLayout(size_hint=(None, 1), spacing=3)
        self._timeline_row.bind(minimum_width=self._timeline_row.setter("width"))
        timeline_scroll.add_widget(self._timeline_row)
        root.add_widget(timeline_scroll)

        # ── Observer log panel ───────────────────────────────────
        root.add_widget(Label(
            text="Observer Log",
            font_size="13sp",
            color=[0.7, 0.7, 0.7, 1],
            size_hint_y=None,
            height=22,
            halign="left",
        ))
        log_scroll = ScrollView(size_hint=(1, None), height=90)
        self._log_label = Label(
            text="",
            font_size="11sp",
            color=[0.5, 0.9, 0.5, 1],
            size_hint_y=None,
            halign="left",
            valign="top",
        )
        self._log_label.bind(
            width=lambda inst, w: setattr(inst, "text_size", (w, None)),
            texture_size=lambda inst, v: setattr(inst, "height", v[1] + 8),
        )
        log_scroll.add_widget(self._log_label)
        root.add_widget(log_scroll)

        # ── Simulate button ─────────────────────────────────────
        sim_btn = Button(
            text="Simulate SAD Message",
            size_hint=(0.5, None),
            height=38,
            pos_hint={"center_x": 0.5},
            font_size="13sp",
            background_color=_hex_to_rgba("#7E57C2"),
            color=[1, 1, 1, 1],
        )
        sim_btn.bind(on_release=self._simulate_message)
        root.add_widget(sim_btn)

        root.add_widget(Widget())  # spacer
        self.add_widget(root)

        # Bind ViewModel properties to UI update methods
        self._vm.bind(message_count=self._on_message_count)
        self._vm.bind(emotion_counts=self._on_emotion_counts)
        self._vm.bind(current_risk=self._on_current_risk)
        self._vm.bind(risk_history=self._on_risk_history)

    # ── Public API ──────────────────────────────────────────────

    def set_memory(self, memory: ConversationMemory) -> None:
        """Inject the Repository and do an initial refresh."""
        self._memory = memory
        self._add_log("[Repository] Memory injected — initial refresh")
        self._vm.refresh(memory)

    def on_pre_enter(self, *_: object) -> None:
        """Refresh dashboard every time the screen is navigated to."""
        if self._memory is not None:
            self._vm.refresh(self._memory)
            self._add_log("[Observer] Screen entered — refreshed from repository")

    # ── ViewModel → UI bindings ─────────────────────────────────

    def _on_message_count(self, _inst: object, value: int) -> None:
        self._card_messages.set_value(str(value))

    def _on_emotion_counts(self, _inst: object, counts: dict[str, int]) -> None:
        total = max(sum(counts.values()), 1)
        for emotion in EmotionLabel:
            bar = self._emotion_bars.get(emotion.value)
            if bar is not None:
                ratio = counts.get(emotion.value, 0) / total
                Animation(size_hint_x=ratio, duration=0.3).start(bar)
        self._add_log(f"[Observer] emotion_counts updated: {counts}")

    def _on_current_risk(self, _inst: object, state: str) -> None:
        self._card_risk.set_value(state)
        color = RISK_COLORS.get(state, RISK_COLORS["calm"])
        anim = Animation(size_hint_y=1.2, duration=0.1) + Animation(size_hint_y=1.0, duration=0.1)
        anim.start(self._card_risk)
        self._add_log(f"[Observer] risk_level updated: {state}")

    def _on_risk_history(self, _inst: object, history: list[str]) -> None:
        user_turns = len([r for r in history if r in RISK_COLORS])
        self._card_turns.set_value(str(user_turns))
        self._rebuild_timeline(history)

    # ── Timeline ────────────────────────────────────────────────

    def _rebuild_timeline(self, history: list[str]) -> None:
        self._timeline_row.clear_widgets()
        for state in history:
            dot = Widget(size_hint=(None, 1), width=18)
            color = RISK_COLORS.get(state, RISK_COLORS["calm"])
            with dot.canvas:
                Color(*_hex_to_rgba(color))
                RoundedRectangle(size=(16, 16), pos=(dot.x + 1, dot.y + 6), radius=[4])
            self._timeline_row.add_widget(dot)

    # ── Simulate ────────────────────────────────────────────────

    def _simulate_message(self, *_: object) -> None:
        if self._memory is None:
            self._add_log("[Simulate] No memory injected — cannot simulate")
            return
        from datetime import datetime
        from safehaven.models import EmotionLabel as EL, Message, RiskLevel
        fake_msg = Message(
            role="user",
            content="[simulated] I've been feeling really sad and hopeless.",
            emotion=EL.SAD,
            risk_level=RiskLevel.MEDIUM,
        )
        self._memory.store_message(fake_msg)
        self._vm.refresh(self._memory)
        self._add_log("[Observer] Simulate triggered — SAD message injected")

    # ── Log ─────────────────────────────────────────────────────

    def _add_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > 20:
            self._log_lines = self._log_lines[-20:]
        self._log_label.text = "\n".join(self._log_lines)

    # ── Navigation ──────────────────────────────────────────────

    def _go_back(self, *_: object) -> None:
        if self.manager is not None:
            self.manager.current = "chat"

    def _upd_bg(self, *_: object) -> None:
        self._bg.size = self.size
        self._bg.pos = self.pos
