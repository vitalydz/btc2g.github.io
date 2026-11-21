"""GitHub publishing jobs for ACC Desktop and headless/cloud runners."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

MARKER = "<!-- acc-job:v1 -->"
REPO_PATTERN = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
SHA_PATTERN = r"[0-9a-f]{40}"


class BridgeError(Exception):
    pass


class PublicationFailed(BridgeError):
    pass


def github_token():
    for name in ("ACC_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    try:
        result = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise BridgeError("Configure ACC_GITHUB_TOKEN, GH_TOKEN or gh auth login; never put a token in an issue.")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BridgeError("GitHub redirected the request; check the configured repository.")


class GitHubClient:
    def __init__(self, repo, token):
        if not re.fullmatch(REPO_PATTERN, repo):
            raise BridgeError("Repository must be owner/repo.")
        self.repo = repo
        self._token = token
        self.opener = build_opener(NoRedirect())

    def request(self, method, path, data=None):
        if not path.startswith("/") or path.startswith("//") or ".." in path:
            raise BridgeError("Invalid GitHub API path.")
        request = Request("https://api.github.com" + path, method=method,
                          data=json.dumps(data).encode("utf-8") if data is not None else None,
                          headers={"Authorization": "Bearer " + self._token,
                                   "Accept": "application/vnd.github+json",
                                   "Content-Type": "application/json",
                                   "User-Agent": "ACC-GitHub-Bridge",
                                   "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with self.opener.open(request, timeout=20) as response:
                raw = response.read()
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            # Never include response bodies or credential-bearing request objects in logs.
            raise BridgeError(f"GitHub HTTP {exc.code}; check permissions, branch rules and request parameters.") from None
        except (URLError, TimeoutError, OSError):
            raise BridgeError("GitHub connection failed; a write may have succeeded. Inspect remote state before retrying.") from None

    def api(self, method, suffix, data=None):
        return self.request(method, f"/repos/{self.repo}{suffix}", data)

    def paged(self, suffix):
        page = 1
        while True:
            separator = "&" if "?" in suffix else "?"
            items = self.api("GET", f"{suffix}{separator}per_page=100&page={page}")
            if not isinstance(items, list):
                raise BridgeError("Unexpected GitHub list response.")
            yield from items
            if len(items) < 100:
                break
            page += 1

    def comment(self, number, body):
        return self.api("POST", f"/issues/{int(number)}/comments", {"body": body})


def parse_job(issue, owner, runner):
    if (issue.get("state") != "open" or "pull_request" in issue or
            issue.get("user", {}).get("login", "").casefold() != owner.casefold() or
            not issue.get("title", "").startswith("ACC: ")):
        return None
    body = issue.get("body") or ""
    if MARKER not in body:
        return None
    blocks = re.findall(r"```json\s*\n(.*?)\n```", body, re.S)
    if not blocks and issue["title"] == "ACC: Connection handshake test":
        return {"action": "handshake", "runner": runner}
    if len(blocks) != 1:
        raise BridgeError("Job requires exactly one fenced JSON object.")
    try:
        job = json.loads(blocks[0])
    except ValueError:
        raise BridgeError("Invalid job JSON.") from None
    if not isinstance(job, dict) or job.get("runner") != runner:
        return None
    action = job.get("action")
    fields = {
        "handshake": {"runner", "action"},
        "push": {"runner", "action", "branch", "sha"},
        "pages": {"runner", "action", "branch", "sha"},
        "workflow": {"runner", "action", "branch", "sha", "workflow", "inputs"},
    }
    if action not in fields or set(job) - fields[action]:
        raise BridgeError("Unsupported action or fields; shell commands are not accepted.")
    if action != "handshake":
        if not re.fullmatch(SHA_PATTERN, str(job.get("sha", ""))):
            raise BridgeError("A complete lowercase 40-character commit SHA is required.")
        branch = job.get("branch", "")
        if (not isinstance(branch, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", branch)
                or ".." in branch or "//" in branch or branch.endswith(("/", ".", ".lock"))):
            raise BridgeError("Invalid branch name.")
    if action == "workflow":
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.ya?ml", str(job.get("workflow", ""))):
            raise BridgeError("Workflow must be a YAML filename.")
        inputs = job.get("inputs", {})
        if (not isinstance(inputs, dict) or len(inputs) > 10 or
                not all(isinstance(k, str) and isinstance(v, str) for k, v in inputs.items())):
            raise BridgeError("Workflow inputs must be a small object of string values.")
    return job


def job_digest(issue, job):
    payload = {"title": issue["title"], "body": issue.get("body"), "job": job}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class StateStore:
    """Process-lifetime OS lock and atomic status journal; crashes never replay writes."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+b")
        self.lock.seek(0, os.SEEK_END)
        if self.lock.tell() == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise BridgeError("Another bridge owns this state journal.") from None
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            if not isinstance(self.data, dict):
                raise ValueError
        except (OSError, ValueError):
            self.close()
            raise BridgeError("Invalid bridge journal. Restore it; do not discard unresolved jobs.") from None

    def save(self):
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def close(self):
        self.lock.close()


class Publisher:
    def __init__(self, client, project, stop, workflows=()):
        self.client, self.project, self.stop = client, Path(project).resolve(), stop
        self.workflows = set(workflows)
        self.started_write = False

    def remote_sha(self, branch):
        return self.client.api("GET", "/git/ref/heads/" + quote(branch, safe=""))["object"]["sha"]

    def git(self, *args):
        if self.stop.is_set():
            raise BridgeError("Stopped.")
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if args[0] == "push":
            auth = base64.b64encode(("x-access-token:" + self.client._token).encode()).decode()
            env.update(GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                       GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + auth,
                       GIT_CONFIG_KEY_1="credential.helper", GIT_CONFIG_VALUE_1="")
        proc = subprocess.Popen(["git", *args], cwd=self.project, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + 120
        try:
            while True:
                if self.stop.is_set() or time.monotonic() >= deadline:
                    raise BridgeError("Git interrupted; verify remote state before a new publishing job.")
                try:
                    stdout, _ = proc.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()
        if proc.returncode:
            raise BridgeError("Git failed. Check the local repository, credentials and branch protection; no force push is used.")
        return stdout.strip()

    def execute(self, job):
        self.started_write = False
        action = job["action"]
        if self.stop.is_set():
            raise BridgeError("Stopped before publication.")
        if action == "push":
            remote = self.git("remote", "get-url", "origin").rstrip("/")
            allowed = {f"https://github.com/{self.client.repo}", f"https://github.com/{self.client.repo}.git",
                       f"git@github.com:{self.client.repo}.git"}
            if remote not in allowed:
                raise BridgeError("origin does not match the configured GitHub repository.")
            if self.git("status", "--porcelain"):
                raise BridgeError("Working tree is dirty. Review and commit changes explicitly before a push job.")
            self.git("check-ref-format", "--branch", job["branch"])
            if self.git("rev-parse", "--verify", "HEAD") != job["sha"]:
                raise BridgeError("Local HEAD changed or does not match the approved SHA.")
            # A canonical URL avoids hidden origin pushurl overrides and never embeds credentials.
            self.started_write = True
            self.git("push", f"https://github.com/{self.client.repo}.git",
                     job["sha"] + ":refs/heads/" + job["branch"])
            if self.remote_sha(job["branch"]) != job["sha"]:
                raise BridgeError("Push could not be verified at the requested remote branch.")
            return "COMPLETED", f"Verified push: {job['sha']} to {job['branch']}"
        if self.remote_sha(job["branch"]) != job["sha"]:
            raise BridgeError("Remote branch no longer matches the approved SHA. Create a new job after review.")
        if action == "pages":
            site = self.client.api("GET", "/pages")
            if site.get("build_type", "legacy") != "legacy" or site.get("source", {}).get("branch") != job["branch"]:
                raise BridgeError("Pages is not branch-based on this branch. Use the configured deployment workflow instead.")
            if self.stop.is_set():
                raise BridgeError("Stopped before requesting Pages build.")
            requested_at = datetime.now(timezone.utc)
            self.started_write = True
            result = self.client.api("POST", "/pages/builds", {})
            build_url = result.get("url", "")
            prefix = f"https://api.github.com/repos/{self.client.repo}/pages/builds/"
            build_id = build_url.removeprefix(prefix)
            if not build_url.startswith(prefix) or not (build_id.isdigit() or build_id == "latest"):
                return "DISPATCHED", "Pages accepted the request; inspect GitHub Pages build status."
            for _ in range(120):
                if self.stop.wait(5):
                    return "DISPATCHED", f"Pages build {build_id} continues remotely; local monitoring stopped."
                build = self.client.api("GET", "/pages/builds/" + build_id)
                if build_id == "latest":
                    try:
                        created = datetime.fromisoformat(build.get("created_at", "").replace("Z", "+00:00"))
                        if created.timestamp() < int(requested_at.timestamp()):
                            continue
                    except (ValueError, TypeError):
                        continue
                    actual_id = build.get("url", "").removeprefix(prefix)
                    if actual_id.isdigit():
                        build_id = actual_id
                if build.get("status") == "errored":
                    raise PublicationFailed(f"Pages build {build_id} failed. Inspect GitHub Pages logs.")
                if build.get("status") == "built":
                    if build.get("commit") != job["sha"]:
                        raise BridgeError("Pages built a different commit; deployment of the approved SHA is not verified.")
                    return "COMPLETED", f"Pages built {job['sha']}; {site.get('html_url', '')}"
            return "DISPATCHED", f"Pages build {build_id} is still pending; no retry was submitted."
        if action == "workflow":
            if job["workflow"] not in self.workflows:
                raise BridgeError("Workflow is not in this runner's configured allowlist.")
            if self.stop.is_set():
                raise BridgeError("Stopped before dispatch.")
            self.started_write = True
            self.client.api("POST", f"/actions/workflows/{job['workflow']}/dispatches",
                            {"ref": job["branch"], "inputs": job.get("inputs", {})})
            return "DISPATCHED", ("Workflow accepted, not verified as deployed: "
                f"https://github.com/{self.client.repo}/actions/workflows/{job['workflow']}")
        raise BridgeError("Unsupported publishing action.")


class Bridge:
    def __init__(self, client, owner, runner, store, publisher, stop, emit=print):
        self.client, self.owner, self.runner = client, owner, runner
        self.store, self.publisher, self.stop, self.emit = store, publisher, stop, emit

    def record(self, number, digest, status, detail=""):
        self.store.data[str(number)] = {"digest": digest, "status": status, "detail": detail, "reported": False}
        self.store.save()
        self.emit(f"#{number} {status}: {detail}")

    def report(self, number):
        row = self.store.data[str(number)]
        if row.get("reported"):
            return
        body = (f"ACC v1.3: {row['status']}\nRunner: {self.runner}\n"
                f"Job SHA256: {row['digest']}\n{row['detail']}")
        exists = any(c.get("body") == body and
                     c.get("user", {}).get("login", "").casefold() == self.owner.casefold()
                     for c in self.client.paged(f"/issues/{number}/comments"))
        if not exists:
            self.client.comment(number, body)
        row["reported"] = True
        self.store.save()

    def poll(self):
        for issue in self.client.paged("/issues?state=open&sort=created&direction=asc"):
            if self.stop.is_set():
                break
            try:
                job = parse_job(issue, self.owner, self.runner)
                if not job:
                    continue
                number, digest = issue["number"], job_digest(issue, job)
                row = self.store.data.get(str(number))
                if row:
                    if row["digest"] != digest and row["status"] == "RECEIVED":
                        self.record(number, digest, "RECEIVED", "Job changed; previous approval invalidated.")
                    self.report(number)
                    continue
                # Remote receipts also prevent replay after loss of local state.
                comments = self.client.paged(f"/issues/{number}/comments")
                executed = any(c.get("user", {}).get("login", "").casefold() == self.owner.casefold()
                    and re.match(r"ACC v1\.3: (RUNNING|COMPLETED|FAILED|DISPATCHED|UNKNOWN)", c.get("body", ""))
                    for c in comments)
                if executed:
                    self.record(number, digest, "UNKNOWN", "Existing remote execution receipt; manual reconciliation required.")
                    continue
                self.record(number, digest, "RECEIVED", "Waiting for approval" if job["action"] != "handshake" else "Connection handshake received; no command executed.")
                self.report(number)
            except BridgeError as exc:
                self.emit(f"Issue #{issue.get('number')}: {exc}")

    def approve(self, number, digest):
        row = self.store.data.get(str(number))
        if not row or row["status"] != "RECEIVED" or row["digest"] != digest:
            raise BridgeError("Job is not pending with this digest; refresh and review it again.")
        issue = self.client.api("GET", f"/issues/{number}")
        job = parse_job(issue, self.owner, self.runner)
        if not job or job_digest(issue, job) != digest:
            raise BridgeError("Job changed, closed or is no longer trusted. Refresh before approving.")
        if self.stop.is_set() or job["action"] == "handshake":
            return
        self.record(number, digest, "RUNNING", "Reserved before execution; do not replay after an interruption.")
        # Fail closed if the remote execution receipt cannot be written.
        self.report(number)
        try:
            status, detail = self.publisher.execute(job)
        except Exception as exc:
            status = "FAILED" if (isinstance(exc, PublicationFailed) or
                                   not self.publisher.started_write) else "UNKNOWN"
            detail = str(exc) if isinstance(exc, BridgeError) else "Publishing interrupted; inspect remote state before issuing a new job."
        self.record(number, digest, status, detail)
        self.report(number)


def state_path(repo, runner):
    name = hashlib.sha256((repo.casefold() + ":" + runner).encode()).hexdigest()[:24]
    return Path.home() / ".acc" / ("bridge-" + name + ".json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--owner", required=True, help="Only this GitHub login may submit jobs")
    parser.add_argument("--runner", required=True, help="Unique route, e.g. win11, vitaly-win10 or cloud")
    parser.add_argument("--project", default=".")
    parser.add_argument("--workflow", action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("poll", help="Receive jobs; execute nothing")
    commands.add_parser("status", help="Read local receipt journal")
    serve = commands.add_parser("serve", help="Poll until stopped; explicit opt-in permits remote publishing")
    serve.add_argument("--interval", type=int, default=30)
    serve.add_argument("--allow-remote", action="store_true")
    approve = commands.add_parser("approve")
    approve.add_argument("number", type=int)
    approve.add_argument("--digest", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("job_file")
    submit.add_argument("--title", default="ACC: Publishing request")
    args = parser.parse_args(argv)
    if not re.fullmatch(REPO_PATTERN, args.repo) or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", args.runner):
        parser.error("Invalid repository or runner")
    store = StateStore(state_path(args.repo, args.runner))
    try:
        if args.command == "status":
            print(json.dumps(store.data, indent=2))
            return
        client = GitHubClient(args.repo, github_token())
        if client.request("GET", "/user")["login"].casefold() != args.owner.casefold():
            raise BridgeError("GitHub credentials must belong to the trusted job author.")
        if args.command == "submit":
            job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
            body = MARKER + "\n```json\n" + json.dumps(job, indent=2) + "\n```"
            issue = {"title": args.title, "body": body, "user": {"login": args.owner}, "state": "open"}
            if not parse_job(issue, args.owner, args.runner):
                raise BridgeError("Job runner or title does not match this request.")
            result = client.api("POST", "/issues", {"title": args.title, "body": body})
            print(result["html_url"])
            return
        stop = threading.Event()
        bridge = Bridge(client, args.owner, args.runner, store,
                        Publisher(client, args.project, stop, args.workflow), stop)
        if args.command == "poll":
            bridge.poll()
        elif args.command == "approve":
            bridge.approve(args.number, args.digest)
        elif args.command == "serve":
            if not 10 <= args.interval <= 3600:
                raise BridgeError("Polling interval must be 10–3600 seconds.")
            try:
                while not stop.is_set():
                    bridge.poll()
                    if args.allow_remote:
                        for number, row in list(store.data.items()):
                            if row["status"] == "RECEIVED":
                                bridge.approve(int(number), row["digest"])
                    stop.wait(args.interval)
            except KeyboardInterrupt:
                stop.set()
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except (BridgeError, OSError, ValueError) as error:
        raise SystemExit(str(error))
