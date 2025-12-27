"""Thin wrapper sử dụng ReasoningEngine trong core để tránh nhân bản logic."""

from __future__ import annotations

from core.reasoning_engine import ReasoningEngine as CoreReasoningEngine


class ReasoningEngine(CoreReasoningEngine):
    """Alias tới core ReasoningEngine cho tầng inference."""

    pass


