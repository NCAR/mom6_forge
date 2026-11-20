"""Shared base for mom6_forge's web widgets.

Each widget has two halves:

* a JavaScript front end (``<widget>.js`` in this folder) that only draws state and turns
  each user action into a message, and
* a Python class that turns each message into ordinary mom6_forge calls.

The contract between them:

* State lives in synced traits. Arrays travel as ``bytes`` traits (raw buffers), never as
  JSON lists.
* The front end sends one message per user action, ``{"op": "<name>", ...}``, with optional
  binary buffers. :meth:`ActionWidget._handle` calls ``self._op_<name>(msg, buffers)``.
* A Python error inside a handler is shown in the widget through the ``errors`` trait; the
  full traceback is kept in ``last_traceback``. A JavaScript error is sent back as
  ``{"op": "js_error"}`` and kept in ``js_errors``. Nothing goes to ``print``, which
  JupyterLab hides in its log console.

The JavaScript is written and maintained by AI. ``_prelude.js`` holds the shared helpers and
the rules every front end follows.
"""

import traceback
from pathlib import Path

import anywidget
import numpy as np
import traitlets

_HERE = Path(__file__).parent


class ActionWidget(anywidget.AnyWidget):
    """Base class: one action channel from the front end, errors shown in the widget."""

    #: File name of the widget's front end, next to this module. Set by each subclass.
    _js_file = ""

    errors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    """Messages from the last action that failed in Python, shown in the widget."""

    js_errors = traitlets.List()
    """Errors the front end reported, as ``{"message", "stack"}`` dicts (Python side only)."""

    last_traceback = traitlets.Unicode("")
    """Full traceback of the last Python error raised by an action."""

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get("_js_file"):
            prelude = (_HERE / "_prelude.js").read_text()
            cls._esm = prelude + "\n" + (_HERE / cls._js_file).read_text()
        super().__init_subclass__(**kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_msg(self._handle)

    def _handle(self, _widget, msg, buffers):
        op = msg.get("op") if isinstance(msg, dict) else None
        if op == "js_error":
            report = {"message": msg.get("message", ""), "stack": msg.get("stack", "")}
            self.js_errors = [*self.js_errors, report]
            return
        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            self.errors = [f"The widget sent an unknown action: {op!r}."]
            return
        self.errors = []
        try:
            handler(msg, list(buffers or []))
        except Exception as exc:  # shown to the person, not swallowed
            self.last_traceback = traceback.format_exc()
            self.errors = [f"{type(exc).__name__}: {exc}"]

    def _act(self, op, buffers=None, **fields):
        """Send an action exactly as the front end would. Used by tests."""
        self._handle(self, {"op": op, **fields}, buffers or [])


def as_bytes(array, dtype):
    """Pack a numpy array into raw bytes for a ``bytes`` trait or a message buffer."""
    return np.ascontiguousarray(array, dtype=dtype).tobytes()


def from_buffer(buffer, dtype):
    """Unpack a message buffer (bytes or memoryview) sent by the front end."""
    return np.frombuffer(bytes(buffer), dtype=dtype)
