# ACC standalone runtime — 2026-11-25

ACC launched the desktop application's bundled codex.exe while inheriting the
parent app's runtime-service bindings. Both shell execution and Windows tooling
failed with `app runtime provisioning service is unavailable; refusing helper fallback`.

The official standalone npm package @openai/codex 0.172.0 is installed locally
under ACC_1_3/runtime, with its complete resources and helpers. ACC now prefers
this complete local package. Run Install_ACC_Runtime.bat with Node/npm installed
to reproduce the installation; binaries are not included in the source archive.

For this standalone child process only, ACC removes four desktop routing variables:
CODEX_APP_RUNTIME_CLI_PATH, CODEX_APP_TOOLS_PIPE_PATH,
CODEX_WINDOWS_REGISTERED_CORE, CODEX_WINDOWS_SANDBOX_PACKAGE_FAMILY.
The inherited permission profile, managed requirements, sandbox mode and normal
Codex authentication remain in effect. ACC still uses approval policy never and
the configured read-only/workspace-write sandbox. No Defender exclusion or
full-access bypass is applied.

An elevated setup was attempted; Windows required elevation, and the UAC launch
ended in cancellation. This is not recorded as successful provisioning. A direct
standalone sandbox test with managed requirements and the workspace permission
profile subsequently read SITES_CONTEXT.md successfully with the corrected child
environment. Model-driven file and browser checks are reported separately in the
GitHub receipts; do not infer browser access from a successful shell read.

Official installation and sandbox guidance:
https://learn.chatgpt.com/docs/enterprise/windows-deployment
https://learn.chatgpt.com/docs/windows/windows-sandbox

Live ACC check: Message 7160869842 in Issue 2 completed. The resumed sites
session read the exact random file probe (not included in its prompt), and the
Windows tool successfully located Chrome. Website/admin data access is a
separate check, not implied by the presence of a browser window.
Regression suite: 93 tests passed before publication.

The follow-up Chrome navigation test (Message 7160905451) was stopped by
Computer Use because it could not determine the current browser URL on Windows
with enough confidence to enforce policy. The agent stopped as instructed.
VitdArt page reading and registration counts are therefore NOT verified.
This is a separate browser policy check, not the previous runtime-service error.
