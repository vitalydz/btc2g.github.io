> Desktop delivery was replaced on 2026-10-03. See [DESKTOP_TARGET_FIX.md](DESKTOP_TARGET_FIX.md) for current targeting, input requirements, retry protection and verification limits. The historical title-matching/clipboard instructions below no longer apply. Restart running ACC instances to load the fix.

# ACC — Agent Codex Controller

## GitHub / Mobile / Codex Cloud

Added 2026-10-30: open **GitHub / Mobile** for a separate owner-filtered publishing bridge.
It supports reviewed-commit push, GitHub Pages builds, and allowlisted workflow dispatches
from GitHub Issues, with local approval by default and an explicit remote-publishing opt-in.
Headless/cloud use does not require Tkinter. See [GITHUB_MOBILE_CLOUD.md](GITHUB_MOBILE_CLOUD.md)
for setup, the mobile GPT Action schema, examples, permissions, and verification limits.

Run `Start_ACC_v1_3.bat` or `python ACC_v1_3.py` on Windows 10/11 with Python and Tkinter installed. Opening ACC does not start the schedule; click START when ready.

The original usage guide is `ACC_v1_2_README.txt`. New installations default to this ACC folder; saved project settings take precedence. Set the project directory to the project you intend to work on.

## Desktop mode

Open the intended Codex conversation and focus its prompt input before starting ACC. Desktop mode sends to the currently selected conversation; the project directory does not select a conversation. The title hint must match one window only. ACC excludes its own windows and reports ambiguous matches instead of choosing the first one.

Desktop submission uses clipboard paste and Enter. Keep the same conversation selected throughout the task. A successful send is not completion. Close or stop any v1.2 controller before starting v1.3; v1.3 prevents a second v1.3 application instance, but cannot control older controllers or manual submissions.

## v1.3 task monitoring

The GUI shows Idle, Sending, Working, Completed, Waiting for reset, Error, or Stopped. "Monitor every" defaults to 30 seconds (1–3600 configurable). The existing "Inspect after send" setting is the initial inspection delay. Existing settings, agent profiles, notes, handoff panel, logs, Days, and Repeat are preserved.

A task records its start time, send time, original handoff timestamp, state, reset time, next attempt, and validated Desktop process/window handle. The task journal is stored beside the existing settings at `~/.acc_agent_codex_controller.task.json`, using atomic replacement. A pending task survives STOP/START and application restarts: START resumes monitoring rather than submitting again. Invalid journals block START because the previous task state cannot be established.

Completion evidence is evaluated in this order:

1. `ACC_HANDOFF.md` has a different nanosecond modification timestamp from the baseline, at or after task start. This is the strongest signal, including a newly created file or unchanged contents with a new timestamp. ACC loads the panel through the main-thread queue and records completion in `ACC_HISTORY.log`.
2. UI Automation observes a recognized working/Stop control followed by two consecutive observations of an idle/input-ready UI. An input-ready window alone does not prove that a submitted task completed.
3. Fresh status-like usage-limit or failure text changes the state to Waiting for reset or Error. Baseline conversation text is ignored, and a visible working control takes precedence over UI error/limit text.

UI inspection uses an optional `pywinauto` helper process restricted to the validated Codex window handle and PID. Each inspection has a timeout and is cancellable. Missing `pywinauto`, inaccessible/unrecognized controls, a disappearing window, or uncertain send delivery causes conservative handoff-only monitoring, without blind resubmissions. Install the optional package with `python -m pip install pywinauto` if desired.

## Schedule and STOP

The configured interval is the delay **after a completed task cycle**, not a periodic send timer. A detected usage limit schedules the parsed reset time plus 20 seconds, or the configured interval when parsing fails. English relative hours/minutes/seconds, local clock times, and basic ISO date/time strings are supported. Unrecognized reset formats fall back to the interval.

Repeat off permits one cycle, including its completion monitoring, but no automatic next cycle or retry. Schedule and target settings are captured at START; stop and restart to apply changes. "Send to Codex" updates notes for the next submission, not the task already running.

STOP wakes waits, cancels ACC's Desktop helper/monitoring, and prevents future submissions. It does not terminate an already-submitted Codex Desktop task. The unresolved task journal remains. Days also bounds monitoring and scheduled waits. CLI mode continues to terminate its own subprocess on STOP; its exit status determines completion or failure. CLI prompts use stdin and output appears when the process finishes/stops.

## Limitations and recovery

UI labels and accessibility exposure vary by Codex version/language. The detector recognizes common English Stop/Send controls, not a documented Codex task-state API. Keep the selected chat unchanged: a window handle is not a conversation identifier. Old transcript text, identical repeated limit messages, or missing accessibility signals can leave ACC waiting conservatively. UI-only completion requires observing the working phase; very short tasks may therefore require a handoff.

The handoff rule deliberately treats any qualifying write as completion, as requested. The prompt should tell Codex to write the handoff only at the end; another process writing that same file can otherwise signal completion early. Without UIA, ACC cannot detect a task started outside ACC before its first submission: ensure the selected chat is idle before the first START. STOP cannot retract an Enter keystroke already delivered.

If delivery failed or a task failed without a handoff, ACC retains the unresolved task rather than guessing it is safe to resend. After verifying in Codex that the task is no longer running, save the project handoff to resolve it and resume monitoring. Do not discard a pending journal while its task may still be running.

## Verification

```powershell
python -m unittest -v
python -m py_compile ACC_v1_3.py acc_task_state.py acc_desktop_probe.py
```

Tests use temporary projects/settings, withdrawn Tk windows, synthetic UI observations, and local dummy subprocesses. They cover timestamps, state transitions, duplicate prevention, STOP/restart, missing inspection, ambiguous delivery, reset scheduling, worker/Tk separation, and helper cancellation. They do not submit prompts to Codex or change real saved settings.

Windows 11 tests and compilation pass. Live Codex UI labels/delivery and Windows 10 (VITALY remote PC) have not been verified in this upgrade. The original v1.2 source and launcher remain available unchanged.

## Fixes, 2026-10-03

- Pass Desktop title hints as data, preserve UTF-8, exclude ACC's own window, and reject ambiguous matches.
- Keep Tk access on the main thread and capture worker configuration before launch.
- Reject blank project paths, non-finite numbers, invalid units/modes, and overflowing schedules.
- Stop silent CLI processes promptly and pass long prompts through stdin.
- Honor Repeat for failed attempts; do not classify successful CLI output as a usage-limit failure merely because it discusses limits.
- Parse relative reset messages with hours, minutes, and seconds.
- Clear stale handoffs when changing projects or deleting a handoff file; keep worker history/logs associated with their original project.
- Recover from non-object settings JSON and malformed agent-list containers.

Local Windows 11 regression checks passed. Windows 10 (VITALY remote PC) and actual Desktop prompt delivery remain unverified.
