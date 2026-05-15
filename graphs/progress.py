"""Lightweight, swappable progress channel.

Anyone (CLI, Streamlit UI, tests) can register a sink to receive structured
progress events as the pipeline runs. Default sink is a no-op so production
tools that don't care can ignore it.

Event shape:
    {
        "phase":  "start" | "done" | "info",
        "label":  human-readable agent / step name,
        "ts":     ISO-8601 UTC timestamp,
    }
"""
from __future__ import annotations
from datetime import datetime
from typing import Callable, Dict, Any, List

# A sink is any callable that accepts a dict event.
Sink = Callable[[Dict[str, Any]], None]

_sinks: List[Sink] = []


def register(sink: Sink) -> Sink:
    """Add a sink. Returns the sink so it can be unregistered later."""
    _sinks.append(sink)
    return sink


def unregister(sink: Sink) -> None:
    if sink in _sinks:
        _sinks.remove(sink)


def reset() -> None:
    _sinks.clear()


def emit(phase: str, label: str, **extra: Any) -> None:
    event = {
        "phase": phase,
        "label": label,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        **extra,
    }
    for sink in list(_sinks):
        try:
            sink(event)
        except Exception:
            # Never let a broken sink kill the pipeline.
            pass


# ---------- Built-in sinks --------------------------------------------------
def console_sink(use_color: bool = True) -> Sink:
    """Pretty-print events to stdout, e.g.

        ▶ Memory layer
        ✓ Memory layer (1.4s)
    """
    def _c(s, code):
        return f"\033[{code}m{s}\033[0m" if use_color else s

    starts: Dict[str, datetime] = {}

    def sink(ev: Dict[str, Any]) -> None:
        label = ev["label"]
        phase = ev["phase"]
        if phase == "start":
            starts[label] = datetime.utcnow()
            print(_c(f"  ▶ {label} ...", "36"), flush=True)
        elif phase == "done":
            t0 = starts.pop(label, None)
            took = (datetime.utcnow() - t0).total_seconds() if t0 else None
            tail = f"  ({took:.1f}s)" if took is not None else ""
            print(_c(f"  ✓ {label}{tail}", "32"), flush=True)
        elif phase == "info":
            print(_c(f"  · {label}", "33"), flush=True)
    return sink


def list_sink(buffer: List[Dict[str, Any]]) -> Sink:
    """Append events into a caller-owned list (used by Streamlit UI)."""
    def sink(ev: Dict[str, Any]) -> None:
        buffer.append(ev)
    return sink
