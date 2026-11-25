"""Vita -> GitHub -> ACC task queue. Only ACC polls; Codex runs actual work."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from acc_github_bridge import BridgeError, GitHubClient, RateLimited, StateStore, github_token

MARKER = "<!-- acc-inbox:v1 -->"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "UNKNOWN", "REJECTED", "WAITING_USER", "ANSWERED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_message(comment, owner, runner):
    if comment.get("user", {}).get("login", "").casefold() != owner.casefold():
        return None
    body = comment.get("body") or ""
    if not body.lstrip().startswith(MARKER):
        return None
    blocks = re.findall(r"```json\s*\n(.*?)\n```", body, re.S)
    if len(blocks) != 1:
        raise BridgeError("Use one fenced JSON object after the acc-inbox marker.")
    try:
        job = json.loads(blocks[0])
    except ValueError:
        raise BridgeError("Invalid inbox JSON.") from None
    if not isinstance(job, dict) or job.get("runner") != runner:
        return None
    action = job.get("action")
    allowed = {"task": {"runner", "action", "target", "prompt"},
               "reply": {"runner", "action", "target", "prompt", "reply_to"},
               "status": {"runner", "action", "target"},
               "cancel": {"runner", "action", "target"}}
    if action not in allowed or set(job) - allowed[action]:
        raise BridgeError("Only task, status and cancel are accepted; no shell/path/session overrides.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(job.get("target", ""))):
        raise BridgeError("A configured target alias is required.")
    if action in {"task", "reply"} and (not isinstance(job.get("prompt"), str) or not 1 <= len(job["prompt"].strip()) <= 20000):
        raise BridgeError("Task prompt must contain 1–20000 characters.")
    if action == "reply" and not re.fullmatch(r"[0-9]{1,30}", str(job.get("reply_to", ""))):
        raise BridgeError("reply_to must identify the message that is waiting for the user.")
    return job


def digest(comment):
    return hashlib.sha256((comment.get("body") or "").encode("utf-8")).hexdigest()


def load_targets(path):
    targets = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(targets, dict) or not targets:
        raise BridgeError("Targets file must map aliases to project/session settings.")
    for alias, target in targets.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", alias) or not isinstance(target, dict):
            raise BridgeError("Invalid target entry.")
        project = Path(target.get("project", "")).expanduser()
        if not target.get("project") or not project.is_absolute() or not project.is_dir():
            raise BridgeError(f"Target {alias}: project must be an existing absolute directory.")
        target["project"] = str(project.resolve())
        session = target.get("session_id")
        if session:
            try:
                target["session_id"] = str(uuid.UUID(session))
            except (ValueError, TypeError):
                raise BridgeError(f"Target {alias}: use a session UUID or null for a dedicated new chat.") from None
        if target.get("sandbox", "workspace-write") not in {"read-only", "workspace-write"}:
            raise BridgeError("Inbox targets cannot disable sandboxing.")
        context = target.get("context", "")
        if not isinstance(context, str) or len(context) > 20000:
            raise BridgeError("Local target context must be text up to 20000 characters.")
    return targets


def codex_environment():
    # Codex keeps its own login, but never inherits the bridge's GitHub credentials.
    return {key: value for key, value in os.environ.items()
            if key.upper() not in {"ACC_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN",
                                   "GITHUB_ENTERPRISE_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS"}
            and not key.upper().startswith(("GIT_CONFIG_", "ACC_BRIDGE_"))}


def redact(text, secrets=()):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{16,})\b",
                  "[REDACTED]", text)


def dialogue_result(text):
    marker = "ACC_QUESTION:"
    if text.strip().startswith(marker):
        question = text.strip()[len(marker):].strip()
        if question:
            return {"state": "WAITING_USER", "result": question}
    return {"state": "COMPLETED", "result": text or "Task completed."}


class CodexExecutor:
    def __init__(self, executable="codex"):
        self.executable = executable
        self.launch_count = 0

    def execute(self, target, session_id, prompt, cancel, notify):
        if cancel.is_set():
            return {"state": "CANCELLED", "result": "Cancelled before launch."}
        with tempfile.TemporaryDirectory(prefix="acc-turn-") as directory:
            output = Path(directory) / "last-message.txt"
            command = [self.executable, "--no-daemon", "--cd", target["project"], "-a", "never", "exec",
                       "--sandbox", target.get("sandbox", "workspace-write")]
            if session_id:
                command += ["resume"]
            command += ["--json", "--output-last-message", str(output)]
            if target.get("skip_git_check", False):
                command += ["--skip-git-repo-check"]
            if session_id:
                command += [session_id]
            command += ["-"]
            instructions = (
                "This is an ACC-delivered task. Do not poll GitHub or wait for further messages. "
                "Complete this task and return a concise result. Do not publish secrets. "
                "You are speaking to the user through Vita, a relay. Answer in the user's language. "
                "If a necessary clarification or approval is missing, end this turn with a final answer "
                "starting exactly ACC_QUESTION: followed by the question. Do not wait or poll; "
                "the user's reply will arrive as another turn in this same session. "
                "Do not push, deploy or publish externally unless this task explicitly requests it.\n\n"
                + ("Local target scope:\n" + target["context"] + "\n\n" if target.get("context") else "")
                + "User message:\n" + prompt)
            with (Path(directory) / "stderr.txt").open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(command, cwd=target["project"], env=codex_environment(),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    start_new_session=(os.name != "nt"))
                self.launch_count += 1
                lines = queue.Queue()

                def read_output():
                    try:
                        for line in proc.stdout:
                            lines.put(line)
                    finally:
                        lines.put(None)

                reader = threading.Thread(target=read_output, daemon=True)
                reader.start()
                done = False
                turn_failed = False
                discovered = session_id
                ended = False
                try:
                    proc.stdin.write(instructions)
                    proc.stdin.close()
                    while not ended:
                        if cancel.is_set():
                            self.terminate(proc)
                            break
                        try:
                            line = lines.get(timeout=0.2)
                        except queue.Empty:
                            if proc.poll() is not None and not reader.is_alive():
                                break
                            continue
                        if line is None:
                            ended = True
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        kind = event.get("type")
                        if kind == "thread.started":
                            discovered = str(uuid.UUID(event["thread_id"]))
                            if session_id and discovered != session_id:
                                raise BridgeError("Codex resumed a different session; execution stopped.")
                            notify("session", discovered)
                        elif kind == "turn.completed":
                            done = True
                        elif kind in {"turn.failed", "error"}:
                            turn_failed = True
                        elif kind == "item.started":
                            notify("progress", "Codex is executing the task")
                    code = proc.wait(timeout=10)
                finally:
                    if proc.poll() is None:
                        self.terminate(proc)
                    reader.join(timeout=3)
                    proc.stdout.close()
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                if cancel.is_set():
                    return {"state": "CANCELLED", "session_id": discovered,
                            "result": "ACC stopped its Codex process. Inspect any partial file changes."}
                text = output.read_text(encoding="utf-8").strip() if output.exists() else ""
                if code == 0 and done and not turn_failed and discovered:
                    return dict(dialogue_result(text), session_id=discovered)
                return {"state": "FAILED" if code != 0 or turn_failed else "UNKNOWN", "session_id": discovered,
                        "result": text or f"Codex did not confirm successful completion (exit {code}). Check local Codex configuration/authentication."}

    @staticmethod
    def terminate(proc):
        if proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class Inbox:
    """Only this coordinator mutates durable state; the execution thread emits events."""
    def __init__(self, client, owner, runner, issue, targets, store, executor=None, auto=False, emit=print):
        self.client, self.owner, self.runner, self.issue = client, owner, runner, int(issue)
        self.targets, self.store, self.executor = targets, store, executor or CodexExecutor()
        self.auto, self.emit = auto, emit
        self.events = queue.Queue()
        self.thread = None
        self.active = None
        self.cancel = threading.Event()
        self.project_lock = None
        self.closed = False
        self.last_error = None
        self.poll_count = 0
        self.messages = store.data.setdefault("messages", {})
        self.sessions = store.data.setdefault("sessions", {})
        identity = {"repo": client.repo, "runner": runner, "issue": self.issue}
        if store.data.get("identity", identity) != identity:
            raise BridgeError("Inbox journal belongs to another repository/runner/issue.")
        store.data["identity"] = identity
        bindings = store.data.setdefault("bindings", {})
        for alias, target in targets.items():
            fingerprint = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
            if alias in bindings and bindings[alias] != fingerprint:
                raise BridgeError(f"Target {alias} changed. Use a new alias to avoid redirecting an existing session or queued task.")
            bindings[alias] = fingerprint
        for row in self.messages.values():
            if row["state"] == "RUNNING":
                row.update(state="UNKNOWN", detail="ACC restarted during this task. It will not be replayed.", reported=False)
        self.store.save()

    def record(self, key, state, detail, **updates):
        row = self.messages[key]
        row.update(state=state, detail=detail, updated_at=now(), reported=False, **updates)
        self.store.save()
        self.emit(f"{key}: {state} — {detail[:140]}")

    def receipt(self, key):
        row = self.messages[key]
        marker = f"<!-- acc-receipt:{self.runner}:{key}:{row['state']} -->"
        body = (f"{marker}\nACC: {row['state']}\nRunner: {self.runner}\nTarget: {row['job']['target']}\n"
                f"Message: {key}\nUpdated: {row['updated_at']}\nSession: {row.get('session_id') or '-'}\n\n"
                + row["detail"][:8000])
        return redact(body, (getattr(self.client, "_token", ""),))

    def flush_reports(self, comments):
        existing = [c.get("body", "") for c in comments
                    if c.get("user", {}).get("login", "").casefold() == self.owner.casefold()]
        for key, row in self.messages.items():
            if row.get("reported"):
                continue
            body = self.receipt(key)
            if body not in existing:
                self.client.comment(self.issue, body)
            row["reported"] = True
            self.store.save()

    def ingest(self, comments):
        for comment in sorted(comments, key=lambda c: c["id"]):
            key = str(comment["id"])
            if key in self.messages:
                continue
            try:
                job = parse_message(comment, self.owner, self.runner)
            except BridgeError as exc:
                self.emit(f"Comment {key} rejected: {exc}")
                continue
            if not job:
                continue
            self.messages[key] = {"job": job, "digest": digest(comment), "state": "RECEIVED",
                                  "created_at": now(), "updated_at": now(), "reported": False, "detail": "Received by ACC"}
            if job["target"] not in self.targets:
                self.record(key, "REJECTED", "Unknown local target alias; configure it in ACC first.")
                continue
            # Never replay an execution after local journal loss.
            prefix = f"<!-- acc-receipt:{self.runner}:{key}:"
            remote = [c.get("body", "") for c in comments if c.get("user", {}).get("login", "").casefold() == self.owner.casefold()
                      and c.get("body", "").startswith(prefix)]
            if any(any(prefix + state + " -->" in body for state in TERMINAL | {"RUNNING"}) for body in remote):
                self.record(key, "UNKNOWN", "Remote execution receipt already exists; no replay.")
                continue
            if job["action"] == "status":
                self.record(key, "COMPLETED", self.status(job["target"]))
            elif job["action"] == "cancel":
                for other, row in list(self.messages.items()):
                    if other != key and row["job"]["target"] == job["target"] and row["state"] in {"QUEUED", "RECEIVED", "WAITING_USER"}:
                        self.record(other, "CANCELLED", "Cancelled by an explicit owner command.")
                if self.active and self.messages[self.active]["job"]["target"] == job["target"]:
                    self.cancel.set()
                self.record(key, "COMPLETED", "Cancellation requested for ACC-owned work; unrelated Codex chats were not touched.")
            else:
                if job["action"] == "reply":
                    parent = self.messages.get(str(job["reply_to"]))
                    if not parent or parent["state"] != "WAITING_USER" or parent["job"]["target"] != job["target"]:
                        self.record(key, "REJECTED", "Reply does not match a question currently waiting on this target.")
                        continue
                state = "QUEUED" if self.auto else "RECEIVED"
                self.record(key, state, "Queued; Codex receives nothing until its turn." if self.auto else "Waiting for local approval.")

    def status(self, target):
        rows = [(key, r) for key, r in self.messages.items()
                if r["job"]["target"] == target and r["job"]["action"] in {"task", "reply"}]
        running = [key for key, r in rows if r["state"] == "RUNNING"]
        waiting = [key for key, r in rows if r["state"] in {"QUEUED", "RECEIVED"}]
        recent = next(((key, r) for key, r in reversed(rows) if r["state"] in TERMINAL), None)
        summary = f"ACC status (no model call): running={running or 'none'}, queued={waiting or 'none'}."
        summary += f"\nSession: {self.sessions.get(target) or self.targets[target].get('session_id') or 'not created yet'}"
        for key, row in rows:
            if row["state"] == "WAITING_USER":
                summary += f"\nWAITING_USER; reply_to={key}: {row['detail'][:1500]}"
        if recent:
            summary += f"\nLast task {recent[0]}: {recent[1]['state']}\n{recent[1]['detail'][:1500]}"
        return summary

    def approve(self, key, expected_digest):
        row = self.messages[key]
        if row["state"] != "RECEIVED" or row["digest"] != expected_digest:
            raise BridgeError("Message revision is no longer pending approval.")
        self.record(key, "QUEUED", "Locally approved; queued for its turn.")

    def drain(self):
        while True:
            try:
                key, kind, value = self.events.get_nowait()
            except queue.Empty:
                return
            row = self.messages[key]
            if kind == "session":
                self.sessions[row["job"]["target"]] = value
                row["session_id"] = value
                self.store.save()
            elif kind == "done":
                if value.get("session_id"):
                    self.sessions[row["job"]["target"]] = value["session_id"]
                self.record(key, value["state"], redact(value["result"], (getattr(self.client, "_token", ""),)),
                            session_id=value.get("session_id"))
                self.active = None
                if self.project_lock:
                    self.project_lock.close()
                    self.project_lock = None

    def dispatch(self):
        if self.active or self.closed:
            return
        if self.thread and self.thread.is_alive():
            return
        for key, row in self.messages.items():
            if row["state"] != "QUEUED":
                continue
            target_name = row["job"]["target"]
            target = self.targets[target_name]
            waiting = {k for k, r in self.messages.items() if r["state"] == "WAITING_USER" and r["job"]["target"] == target_name}
            if row["job"]["action"] == "reply":
                if str(row["job"]["reply_to"]) not in waiting:
                    self.record(key, "REJECTED", "Question has already been answered or cancelled.")
                    continue
            elif waiting:
                continue
            if any(other["state"] == "UNKNOWN" and other["job"]["action"] in {"task", "reply"}
                   and (other["job"]["target"] == target_name or
                        os.path.normcase(self.targets.get(other["job"]["target"], {}).get("project", ""))
                        == os.path.normcase(target["project"])) for other in self.messages.values()):
                continue
            current = self.client.api("GET", f"/issues/comments/{key}")
            if (digest(current) != row["digest"] or parse_message(current, self.owner, self.runner) != row["job"]):
                self.record(key, "CANCELLED", "Message edited before execution. Submit a new comment.")
                continue
            project_id = hashlib.sha256(os.path.normcase(target["project"]).encode()).hexdigest()[:24]
            try:
                self.project_lock = StateStore(Path.home() / ".acc" / f"project-{project_id}.json")
            except BridgeError:
                continue
            self.record(key, "RUNNING", "Codex task is starting; ACC continues polling GitHub independently.")
            # Durable remote reservation must succeed before the process is launched.
            try:
                self.client.comment(self.issue, self.receipt(key))
                row["reported"] = True
                self.store.save()
            except Exception:
                self.record(key, "UNKNOWN", "Could not confirm remote reservation. Task not launched; no automatic replay.")
                self.project_lock.close()
                self.project_lock = None
                raise
            self.active = key
            if row["job"]["action"] == "reply":
                parent_key = str(row["job"]["reply_to"])
                self.record(parent_key, "ANSWERED", self.messages[parent_key]["detail"])
            self.cancel = threading.Event()
            session = self.sessions.get(target_name) or target.get("session_id")

            def work(key=key, target=dict(target), session=session, prompt=row["job"]["prompt"]):
                try:
                    result = self.executor.execute(target, session, prompt, self.cancel,
                        lambda kind, value: self.events.put((key, kind, value)))
                except Exception:
                    result = {"state": "UNKNOWN", "session_id": session,
                              "result": "Codex launch/execution interrupted. Inspect local state before continuing this target."}
                self.events.put((key, "done", result))

            self.thread = threading.Thread(target=work, daemon=True)
            self.thread.start()
            return

    def tick(self):
        self.drain()
        issue = self.client.request("GET", f"/repos/{self.client.repo}/issues/{self.issue}", cached=True)
        if issue.get("state") != "open":
            self.closed = True
            return
        comments = list(self.client.inbox_comments(self.issue))
        self.poll_count += 1
        self.ingest(comments)
        self.flush_reports(comments)
        self.dispatch()
        self.store.data["metrics"] = {"polls": self.poll_count,
                                      "model_launches": getattr(self.executor, "launch_count", 0),
                                      "last_poll_at": now()}
        self.store.save()

    def serve(self, stop, interval=5, commands=None, stop_path=None):
        next_poll = 0.0
        try:
            while not stop.is_set():
                if stop_path and stop_path.exists():
                    stop_path.unlink()
                    stop.set()
                    break
                self.drain()
                if commands:
                    try:
                        key, expected = commands.get_nowait()
                        self.approve(key, expected)
                    except queue.Empty:
                        pass
                if time.monotonic() >= next_poll:
                    delay = interval
                    try:
                        self.tick()
                        self.last_error = None
                    except BridgeError as exc:
                        delay = exc.seconds if isinstance(exc, RateLimited) else max(15, interval)
                        if str(exc) != self.last_error:
                            self.emit(str(exc))
                        self.last_error = str(exc)
                    next_poll = time.monotonic() + delay
                stop.wait(0.1)
        finally:
            self.closed = True
            self.cancel.set()
            if self.thread:
                self.thread.join(timeout=20)
            self.drain()
            if self.active:
                self.record(self.active, "UNKNOWN", "Stopped without confirmed process termination; review before continuing.")
            if self.project_lock:
                self.project_lock.close()
                self.project_lock = None
            try:
                self.flush_reports([])
            except BridgeError:
                self.emit("Final receipts remain in the journal for the next connection.")


def inbox_state_path(repo, runner, issue):
    key = hashlib.sha256(f"{repo.lower()}:{runner}:{issue}".encode()).hexdigest()[:24]
    return Path.home() / ".acc" / f"inbox-{key}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--runner", default="win11")
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--targets")
    parser.add_argument("--stop", action="store_true", help="Request a graceful stop of this local CLI receiver")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--allow-tasks", action="store_true")
    parser.add_argument("--once", action="store_true", help="Receive/status only; never launch a task")
    args = parser.parse_args(argv)
    if not 5 <= args.interval <= 3600 or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", args.runner):
        parser.error("Use a safe runner alias and an interval of 5–3600 seconds")
    stop_path = inbox_state_path(args.repo, args.runner, args.issue).with_suffix(".stop")
    if args.stop:
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        stop_path.write_text(now(), encoding="utf-8")
        print("Stop requested for the local receiver. Active ACC-owned work will be cancelled.")
        return
    if not args.targets:
        parser.error("--targets is required unless --stop is used")
    client = GitHubClient(args.repo, github_token())
    if client.request("GET", "/user")["login"].casefold() != args.owner.casefold():
        raise BridgeError("GitHub authentication must match the trusted author.")
    targets = load_targets(args.targets)
    store = StateStore(inbox_state_path(args.repo, args.runner, args.issue))
    try:
        stop_path.unlink(missing_ok=True)
        inbox = Inbox(client, args.owner, args.runner, args.issue, targets, store, auto=args.allow_tasks and not args.once)
        if args.once:
            inbox.closed = True
            inbox.tick()
        else:
            stop = threading.Event()
            try:
                inbox.serve(stop, args.interval, stop_path=stop_path)
            except KeyboardInterrupt:
                stop.set()
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except (BridgeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc))
