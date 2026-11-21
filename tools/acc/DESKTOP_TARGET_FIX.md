# ACC Desktop target fix — 2026-10-03

## Findings

Process 6820 is ApplicationFrameHost.exe, with title Settings. It is not Codex.
The installed Codex GUI is ChatGPT.exe inside the OpenAI.Codex WindowsApps package.
The codex.exe processes under the versioned bin directory are helpers, not the GUI.

The former delivery path selected a process using MainWindowTitle, activated by PID
using WScript.Shell.AppActivate, and issued global Ctrl+V and Enter after fixed sleeps.
It never verified the foreground HWND or the composer before those keystrokes.
Test Codex only searched for the window, so it did not test the actual activation path.

The source revision inspected already rejected empty title hints and multiple title
matches. Therefore the exact historical Settings selection with the hint Codex cannot
be reproduced from this revision alone. An older running controller or changing
MainWindowTitle is possible; neither is established by the log. The confirmed defect
is the lack of application identity and destination/input verification. The currently
running ACC process loads ACC_v1_3.py and must be restarted to load these edits.

## New behavior

- Enumerate top-level windows. Validate process executable, installation/package path,
  PID, HWND and process creation time. Exclude ACC, Settings, helpers and other apps.
- Ignore the old title hint. Ask for a concrete window when several Codex windows exist.
  Preserve that selection within the controller session. A stale selection fails closed;
  it does not silently move to another window or conversation.
- Restore minimized windows and verify foreground activation. Test Codex and real
  delivery use the same preparation path; Test never writes or submits.
- Require the user-focused, empty, enabled composer with a recognized accessible name,
  a writable UIA ValuePattern, and one recognized Send button in the same window.
  Unknown accessibility layouts, existing drafts and busy Codex fail closed.
- Use addressed UIA SetValue/Invoke operations, never clipboard paste, SendKeys or Enter.
  Recheck identity, foreground and controls before writing and before submission.
  If focus changes after writing, the draft stays in Codex and submission is blocked.
- Never navigate projects, conversations, tabs or settings.
- Report WINDOW_NOT_FOUND, AMBIGUOUS_WINDOW, ACTIVATION_FAILED, INPUT_UNAVAILABLE,
  USAGE_LIMIT and DELIVERY_PENDING/DELIVERY_UNCONFIRMED separately. Activation failure
  never implies an account limit. Preflight limit detection only considers an explicit
  accessible StatusBar, not old conversation text. v1.3 retains its task-state monitor.
- Reserve delivery in ~/.acc_codex_delivery.json before writing. This exclusive durable
  journal contains only process/window IDs, an operation ID and a message hash. Retry,
  restart or a second controller cannot replay an unresolved delivery.
- v1.3 clears the reservation only after matching task completion. v1.2 stops after a
  Desktop delivery because it has no reliable completion monitor. The user can resolve
  a reviewed delivery with the new button after stopping ACC; this also clears selection.
- Diagnostic logs contain target identity, activation result, error and next-attempt
  policy. Desktop message bodies and UI snapshots are no longer appended to v1.2 logs.
  Existing historical logs are preserved.

## Files

ACC_v1_2.py, ACC_v1_3.py, acc_desktop_target.py, acc_desktop_probe.py,
test_acc.py, test_desktop_target.py. The ACC_1_3 distribution receives the same shared
target/probe implementation and v1.3 source. Original files are backed up separately.

## Verification

48 automated tests pass under the installed Windows Python 3.12. Python compilation
passes. Tests cover Settings and ACC foreground, minimized restore-before-activation,
multiple-window selection/cancellation, closed Codex, stale/reused HWND and PID,
inaccessible input, focus loss before/after writing, retry/restart deduplication,
STOP, no message content in logs, and existing v1.3 task-state regressions.

Window/focus scenarios use fake backends, not real desktop input. Native UI control
was unavailable in this session. Actual Codex accessibility names/ValuePattern support
and real Win11/Win10 activation remain unverified. If the composer is not exposed as
expected, INPUT_UNAVAILABLE intentionally blocks delivery; no keyboard fallback exists.

Restart ACC, open the intended conversation, focus its empty composer, then click
Test Codex. A successful test identifies the process and HWND without sending anything.
Do not resolve an uncertain delivery until checking the intended conversation.
