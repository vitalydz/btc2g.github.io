# Implemented publishing bridge — 2026-10-30

Source is included under [tools/acc](../tools/acc). This integrates the installed ACC v1.3 and preserves its Desktop targeting and task-state monitoring fixes. It is not the missing historical ChatGPT ZIP.

- Open ACC's **GitHub / Mobile** panel, or use `tools/acc/acc_github_bridge.py` headlessly in Codex Cloud.
- Owner-filtered, runner-routed Issues support handshake, exact-commit push, branch-based Pages build, and allowlisted workflow dispatch.
- Receive-only/local approval is the default. Remote publishing requires an explicit session opt-in.
- The bridge does not execute arbitrary Codex prompts or shell commands from issue bodies. Existing local Desktop/CLI modes remain available.
- Persistent receipts, content-digest approvals, process locks and remote status comments prevent routine replay. One active executor per runner ID is required; this is not a distributed exactly-once queue.
- GitHub credentials stay out of saved settings, issues and command arguments. No token is included in this source.
- Workflow dispatch is reported as DISPATCHED, not successful deployment.
- Setup, mobile GPT Action schema and cloud commands: [GITHUB_MOBILE_CLOUD.md](../tools/acc/GITHUB_MOBILE_CLOUD.md).
- Windows 11 validation: 70 tests in the development folder, 59 in the distributable v1.3 folder, and Python compilation passed. Cloud-compatible suite: `cd tools/acc && python -m unittest test_acc_bridge -v`.
- Live publishing, home-PC handshake, Windows 10 and cloud execution remain unverified. No live site deployment or main-branch merge was performed.

## Historical specification (2026-10-28)

The following is historical context. Its missing-source and arbitrary Codex-runner statements are superseded by the implementation above.

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
