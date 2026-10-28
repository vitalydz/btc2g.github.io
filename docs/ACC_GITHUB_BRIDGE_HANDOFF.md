# ACC v1.3 GitHub Bridge — handoff (2026-10-28)

ACC (Agent Codex Controller) is a Windows/Tkinter tool originally built as ACC v1.1 with Start/Stop, repeat intervals, run-days, agent selection and local handoff files. ACC v1.3 adds a **separate** GitHub Issue bridge.

## Code artifact
The tested standalone code bundle was created in ChatGPT as `ACC_v1_3_GitHub_Bridge.zip` (ACC_v1_3.py, acc_github_bridge.py, test_acc_bridge.py, Start_ACC_v1_3.bat and README.md). This document is a coordination record, **not a claim that the Python source is committed to GitHub**. Codex must receive the source bundle from the project owner before integrating it.

## GitHub bridge protocol
- Repository: `vitalydz/btc2g.github.io`.
- Poll issue list, default every 30 sec.
- Eligible jobs: open issue authored by `vitalydz`, title starts `ACC: `, body contains `<!-- acc-job:v1 -->`.
- On first receipt, post comment `ACC v1.3: RECEIVED`. Do not execute before explicit local approval by default.
- On approved execution, post `RUNNING`, and then `COMPLETED` or `FAILED`.
- Persist local status to avoid duplicate execution after reboot.
- GitHub access token comes from environment, a local in-memory password dialog, or `gh auth token`. Never store it in project source or pass token env variables into Codex.
- Job runner invokes `codex exec --sandbox workspace-write` in selected folder. **This is Codex CLI, not remote control of the Codex Desktop window.**
- Existing ACC v1.2 Desktop-specific code is not present in the recovered source: preserve the installed v1.2 separately until verified integration is possible.
- No silent automatic push/deployment, no secret leakage.

## Handshake acceptance
1. Owner installs bridge on home Windows PC and configures local GitHub authentication.
2. Start Bridge without auto-run.
3. Issue titled `ACC: Connection handshake test` with exact marker receives `ACC v1.3: RECEIVED` from running PC.
4. Only after confirmation and optional deliberate local approval, a test Codex CLI task may be run.
5. Report issue/commit references from actual runs, not planned behavior.

## Offline test status
11 Python unittest checks for issue trust filtering, idempotency, status reporting, token stripping and process boundaries: PASS as of 2026-10-28 (local container). Live home-PC integration is NOT tested.
