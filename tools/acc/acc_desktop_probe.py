"""Read only the validated Codex window in a bounded, disposable UIA process."""

import json
import sys


def inspect_window(pid, handle):
    try:
        from acc_desktop_target import WinBackend, is_codex
        window_identity = WinBackend().describe(handle)
        if window_identity.pid != pid or not is_codex(window_identity):
            return {"detail": "WINDOW_NOT_FOUND: saved HWND no longer belongs to Codex."}
    except Exception:
        return {"detail": "WINDOW_NOT_FOUND: saved Codex window cannot be validated."}
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"detail": "pywinauto is unavailable; monitoring ACC_HANDOFF.md only. No timed resubmission."}
    try:
        window = Desktop(backend="uia").window(handle=handle).wrapper_object()
        if window.process_id() != pid or not window.is_visible():
            return {"detail": "Validated Codex window is unavailable; awaiting handoff without resubmitting."}
        working = False
        editable = False
        send_control = False
        messages = []
        for control in window.descendants():
            if not control.is_visible():
                continue
            name = (control.window_text() or "").strip()
            kind = control.element_info.control_type
            if kind == "Button":
                normalized = name.casefold()
                if normalized in {"stop", "stop generating", "stop generation", "stop response", "stop task", "interrupt"} and control.is_enabled():
                    working = True
                if normalized in {"send", "send message", "send prompt", "submit"}:
                    send_control = True
            if kind == "Edit" and control.is_enabled():
                editable = True
            if kind in {"Text", "StatusBar"} and name:
                messages.append(name)
        return {"available": True, "working": working,
                "input_ready": editable and send_control and not working,
                "messages": messages,
                "detail": "" if (working or editable) else
                "Codex UI controls are not recognized; awaiting handoff or a recognized task state."}
    except Exception:
        return {"detail": "Codex UI inspection is unavailable; awaiting handoff without resubmitting."}


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    result = inspect_window(int(request["pid"]), int(request["handle"]))
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
