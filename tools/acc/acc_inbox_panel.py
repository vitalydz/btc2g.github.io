"""Desktop controls for the independent Vita/Codex inbox."""
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from acc_codex_inbox import Inbox, inbox_state_path, load_targets
from acc_github_bridge import BridgeError, GitHubClient, StateStore, github_token


class InboxPanel:
    def __init__(self, app, path):
        self.app, self.path = app, Path(path)
        self.thread = None
        self.closed = False
        self.events, self.commands = queue.Queue(), queue.Queue()
        self.stop = threading.Event()
        self.window = tk.Toplevel(app.root)
        self.window.title("ACC — Vita / Codex inbox")
        self.window.geometry("1000x750")
        try:
            settings = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        self.vars = {}
        form = ttk.Frame(self.window, padding=10)
        form.pack(fill="x")
        for i, (key, label, default) in enumerate([
            ("repo", "Repository", "vitalydz/btc2g.github.io"), ("owner", "Trusted GitHub author", "vitalydz"),
            ("runner", "Runner", "win11"), ("issue", "Inbox issue / PR number", ""),
            ("interval", "ACC poll interval (seconds)", "5")]):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w")
            self.vars[key] = tk.StringVar(value=settings.get(key, default))
            ttk.Entry(form, textvariable=self.vars[key]).grid(row=i, column=1, sticky="ew", padx=10, pady=2)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Token (memory only; blank uses environment, gh or Windows Git credentials)").grid(row=5, column=0, sticky="w")
        self.token = tk.StringVar()
        ttk.Entry(form, textvariable=self.token, show="*").grid(row=5, column=1, sticky="ew", padx=10)
        ttk.Label(self.window, text="Local target aliases → project and session UUID. null creates a dedicated chat on the first task.").pack(anchor="w", padx=10)
        self.targets = tk.Text(self.window, height=7, wrap="none")
        self.targets.pack(fill="x", padx=10, pady=5)
        initial = settings.get("targets", {"acc": {"project": app.project_var.get(), "session_id": None, "sandbox": "workspace-write"}})
        self.targets.insert("1.0", json.dumps(initial, indent=2))
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.window, text="Allow Vita's trusted tasks for this session (status checks never invoke Codex)",
                        variable=self.auto).pack(anchor="w", padx=10)
        buttons = ttk.Frame(self.window, padding=10)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="Start inbox", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(buttons, text="Stop inbox", command=self.stop_inbox).pack(side="left", padx=5)
        ttk.Button(buttons, text="Approve selected", command=self.approve).pack(side="left", padx=5)
        self.status = tk.StringVar(value="Stopped")
        ttk.Label(buttons, textvariable=self.status).pack(side="right")
        self.rows = {}
        self.jobs = ttk.Treeview(self.window, columns=("id", "target", "state"), show="headings", height=6)
        for name in ("id", "target", "state"):
            self.jobs.heading(name, text=name.title())
        self.jobs.pack(fill="both", expand=True, padx=10)
        self.log = tk.Text(self.window, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=8)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.after_id = None
        self.pump()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        publisher = getattr(self.app, "github_panel", None)
        if (self.app.worker and self.app.worker.is_alive()) or (publisher and publisher.thread and publisher.thread.is_alive()):
            messagebox.showerror("ACC", "Stop the local schedule and publishing bridge first.", parent=self.window)
            return
        config = {key: var.get().strip() for key, var in self.vars.items()}
        try:
            if not 5 <= int(config["interval"]) <= 3600 or int(config["issue"]) <= 0:
                raise ValueError("Use a positive issue number and a 5–3600 second interval.")
            config["targets"] = json.loads(self.targets.get("1.0", "end-1c"))
            targets_path = self.path.with_suffix(".targets.json")
            targets_path.write_text(json.dumps(config["targets"], indent=2), encoding="utf-8")
            targets = load_targets(targets_path)
            self.path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except (ValueError, OSError, BridgeError) as exc:
            messagebox.showerror("ACC", str(exc), parent=self.window)
            return
        self.stop.clear()
        self.commands = queue.Queue()
        self.start_button.configure(state="disabled")
        self.status.set("Connecting")
        self.thread = threading.Thread(target=self.run,
            args=(config, targets, self.token.get().strip(), self.auto.get()), daemon=False)
        self.thread.start()

    def run(self, config, targets, token, auto):
        store = None
        try:
            client = GitHubClient(config["repo"], token or github_token())
            if client.request("GET", "/user")["login"].casefold() != config["owner"].casefold():
                raise BridgeError("Token owner must match the trusted author.")
            store = StateStore(inbox_state_path(config["repo"], config["runner"], int(config["issue"])))

            def emit(message):
                self.events.put(("log", message))
                self.events.put(("rows", json.loads(json.dumps(store.data.get("messages", {})))))

            inbox = Inbox(client, config["owner"], config["runner"], int(config["issue"]), targets, store, auto=auto, emit=emit)
            self.events.put(("status", "Listening — tasks enabled" if auto else "Listening — approval required"))
            inbox.serve(self.stop, int(config["interval"]), self.commands)
        except Exception as exc:
            self.events.put(("log", str(exc) if isinstance(exc, (BridgeError, ValueError)) else "Inbox stopped after an unexpected error; inspect its journal."))
        finally:
            if store:
                store.close()
            self.events.put(("stopped", None))

    def approve(self):
        selected = self.jobs.selection()
        if not selected:
            return
        key = selected[0]
        row = self.rows[key]
        if row["state"] != "RECEIVED":
            return
        if messagebox.askyesno("ACC — approve task", f"Target: {row['job']['target']}\n\n{row['job'].get('prompt', '')}", parent=self.window):
            self.commands.put((key, row["digest"]))

    def stop_inbox(self):
        self.stop.set()
        self.status.set("Stopping owned task and receiver")

    def pump(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "rows":
                    self.rows = data
                    for key, row in data.items():
                        values = (key, row["job"]["target"], row["state"])
                        if self.jobs.exists(key):
                            self.jobs.item(key, values=values)
                        else:
                            self.jobs.insert("", "end", iid=key, values=values)
                elif kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", data + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "status":
                    self.status.set(data)
                elif kind == "stopped":
                    self.status.set("Stopped")
                    self.start_button.configure(state="normal")
        except queue.Empty:
            pass
        if not self.closed:
            self.after_id = self.window.after(150, self.pump)

    def close(self):
        self.stop_inbox()
        self.closed = True
        if self.after_id:
            self.window.after_cancel(self.after_id)
        self.window.destroy()
