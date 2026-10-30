"""Conservative Desktop task tracking, independent of Tkinter and UI Automation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import re
import time


class CodexState(str, Enum):
    IDLE = "Idle"
    SENDING = "Sending"
    WORKING = "Working"
    COMPLETED = "Completed"
    WAITING = "Waiting for reset"
    ERROR = "Error"
    STOPPED = "Stopped"


@dataclass
class DesktopSnapshot:
    available: bool = False
    working: bool = False
    input_ready: bool = False
    messages: tuple[str, ...] = ()
    detail: str = ""


def handoff_mtime(path):
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return None


@dataclass
class CurrentTask:
    project: str
    task_start_time: datetime = field(default_factory=datetime.now)
    task_start_ns: int = field(default_factory=time.time_ns)
    prompt_sent_time: datetime | None = None
    handoff_mtime_before: int | None = None
    state: CodexState = CodexState.IDLE
    parsed_reset_time: datetime | None = None
    next_attempt_time: datetime | None = None
    pending: bool = True
    window_pid: int | None = None
    window_handle: int | None = None
    observed_working: bool = False
    idle_polls: int = 0
    baseline_messages: tuple[str, ...] = ()

    def new_handoff(self, mtime):
        return (mtime is not None and mtime != self.handoff_mtime_before
                and mtime >= self.task_start_ns)

    def observe(self, snapshot, mtime):
        """Handoff wins; UI-only completion requires an observed busy-to-idle transition."""
        if self.new_handoff(mtime):
            return CodexState.COMPLETED, "New ACC_HANDOFF.md detected - task completed"
        if not snapshot.available:
            self.idle_polls = 0
            return CodexState.WORKING, "Awaiting completion evidence"
        if snapshot.available and snapshot.working:
            self.observed_working = True
            self.idle_polls = 0
            return CodexState.WORKING, "Codex is still working"
        fresh = [message for message in snapshot.messages
                 if message not in self.baseline_messages]
        for message in fresh:
            # Match status-like messages, not arbitrary mentions in old conversation text.
            if re.match(r"^(?:(?:you['’]ve|you have) (?:hit|reached) (?:your |the )?)?"
                        r"(?:usage limit|rate limit|quota exceeded)|^try again later|^resets? (?:at|in)\b",
                        message.strip(), re.I):
                return CodexState.WAITING, "\n".join(fresh)
            if re.match(r"^(?:something went wrong|task failed|an error occurred|connection failed)\b",
                        message.strip(), re.I):
                return CodexState.ERROR, message
        if snapshot.available and snapshot.input_ready and self.observed_working:
            self.idle_polls += 1
            if self.idle_polls >= 2:
                return CodexState.COMPLETED, "Codex UI returned to idle after working"
        else:
            self.idle_polls = 0
        return CodexState.WORKING, "Awaiting completion evidence"

    def to_dict(self):
        data = asdict(self)
        data["state"] = self.state.value
        for key in ("task_start_time", "prompt_sent_time", "parsed_reset_time", "next_attempt_time"):
            data[key] = data[key].isoformat() if data[key] else None
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("task_start_time", "prompt_sent_time", "parsed_reset_time", "next_attempt_time"):
            data[key] = datetime.fromisoformat(data[key]) if data.get(key) else None
        data["state"] = CodexState(data["state"])
        data["baseline_messages"] = tuple(data.get("baseline_messages", ()))
        return cls(**data)
