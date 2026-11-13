"""Tk front end for the publishing bridge; all network work stays off the UI thread."""
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

from acc_github_bridge import (Bridge, BridgeError, GitHubClient, Publisher, StateStore,
                               github_token, state_path)


class GitHubPanel:
    def __init__(self, app, settings_path):
        self.app = app
        self.settings_path = Path(settings_path)
        self.window = tk.Toplevel(app.root)
        self.window.title("ACC — GitHub / Mobile / Cloud")
        self.window.geometry("950x670")
        self.stop = threading.Event()
        self.thread = None
        self.events = queue.Queue()
        self.commands = queue.Queue()
        self.rows = {}
        self.after_id = None
        self.closed = False
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        self.vars = {}
        fields = [("repo", "GitHub repository", "vitalydz/btc2g.github.io"),
                  ("owner", "Trusted issue author", "vitalydz"),
                  ("runner", "Runner ID (unique per PC/cloud)", "win11"),
                  ("project", "Local Git repository", app.project_var.get()),
                  ("workflows", "Allowed workflow filenames (comma separated)", ""),
                  ("interval", "Poll interval (seconds)", "30")]
        form = ttk.Frame(self.window, padding=12)
        form.pack(fill="x")
        for row, (key, label, default) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=settings.get(key, default))
            self.vars[key] = var
            ttk.Entry(form, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="GitHub token (memory only; blank uses environment / gh)").grid(row=6, column=0, sticky="w")
        self.token = tk.StringVar()
        ttk.Entry(form, textvariable=self.token, show="*").grid(row=6, column=1, sticky="ew", padx=8)
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Allow remote publishing from this trusted author for this session",
                        variable=self.auto).grid(row=7, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(form, text="Default: receive only. Push sends an existing reviewed commit; Pages/workflow requests are separate.").grid(row=8, column=0, columnspan=2, sticky="w")
        buttons = ttk.Frame(self.window, padding=(12, 0))
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="Start bridge", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(buttons, text="Stop bridge", command=self.stop_bridge).pack(side="left", padx=5)
        ttk.Button(buttons, text="Review / approve selected", command=self.approve).pack(side="left", padx=5)
        ttk.Button(buttons, text="Open selected issue", command=self.open_issue).pack(side="left")
        self.status = tk.StringVar(value="Stopped")
        ttk.Label(buttons, textvariable=self.status).pack(side="right")
        self.jobs = ttk.Treeview(self.window, columns=("number", "state", "detail"), show="headings", height=8)
        for name, width in (("number", 70), ("state", 120), ("detail", 650)):
            self.jobs.heading(name, text=name.title())
            self.jobs.column(name, width=width)
        self.jobs.pack(fill="both", expand=True, padx=12, pady=8)
        self.log = tk.Text(self.window, height=8, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.pump()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        if self.app.worker and self.app.worker.is_alive():
            messagebox.showerror("ACC", "Stop the local Codex schedule before starting the publishing bridge.", parent=self.window)
            return
        config = {key: var.get().strip() for key, var in self.vars.items()}
        try:
            import re
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", config["runner"]):
                raise ValueError("Invalid runner ID")
            interval = int(config["interval"])
            if not 10 <= interval <= 3600:
                raise ValueError("Polling interval must be 10–3600 seconds")
            if not Path(config["project"]).is_dir():
                raise ValueError("Local project folder does not exist")
            self.settings_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        except (ValueError, OSError) as exc:
            messagebox.showerror("ACC", str(exc), parent=self.window)
            return
        self.config = config
        token, auto = self.token.get().strip(), self.auto.get()
        self.stop.clear()
        self.commands = queue.Queue()
        self.start_button.configure(state="disabled")
        self.status.set("Starting")
        self.thread = threading.Thread(target=self.run, args=(config, token, auto, interval), daemon=True)
        self.thread.start()

    def run(self, config, token, auto, interval):
        store = None
        try:
            client = GitHubClient(config["repo"], token or github_token())
            identity = client.request("GET", "/user")["login"]
            if identity.casefold() != config["owner"].casefold():
                raise BridgeError("Use credentials belonging to the configured trusted author.")
            store = StateStore(state_path(config["repo"], config["runner"]))
            publisher = Publisher(client, config["project"], self.stop,
                                  [x.strip() for x in config["workflows"].split(",") if x.strip()])
            bridge = Bridge(client, config["owner"], config["runner"], store, publisher,
                            self.stop, lambda text: self.events.put(("log", text)))
            self.events.put(("status", "Remote publishing enabled" if auto else "Receive only"))
            while not self.stop.is_set():
                bridge.poll()
                self.events.put(("rows", json.loads(json.dumps(store.data))))
                if auto:
                    for number, row in list(store.data.items()):
                        if row["status"] == "RECEIVED" and not self.stop.is_set():
                            bridge.approve(int(number), row["digest"])
                # Approvals wake this queue immediately instead of waiting for the next poll.
                try:
                    command = self.commands.get(timeout=interval)
                    if command is not None and not self.stop.is_set():
                        number, digest = command
                        bridge.approve(number, digest)
                except queue.Empty:
                    pass
                self.events.put(("rows", json.loads(json.dumps(store.data))))
        except Exception as exc:
            detail = str(exc) if isinstance(exc, (BridgeError, ValueError)) else "Bridge stopped after an unexpected error; inspect the journal before retrying."
            self.events.put(("log", detail))
        finally:
            if store:
                store.close()
            self.events.put(("stopped", None))

    def approve(self):
        selected = self.jobs.selection()
        if not selected or not self.thread or not self.thread.is_alive():
            return
        number = selected[0]
        row = self.rows[number]
        if row["status"] != "RECEIVED":
            return
        self.open_issue()
        if messagebox.askyesno("ACC — approve publication",
            f"Review issue #{number} in GitHub. Execute exactly this revision?\nSHA256: {row['digest']}\n"
            "A changed issue will be rejected. No automatic commit or force push.", parent=self.window):
            self.commands.put((int(number), row["digest"]))

    def open_issue(self):
        selected = self.jobs.selection()
        if selected:
            repo = self.config["repo"] if hasattr(self, "config") else self.vars["repo"].get()
            webbrowser.open(f"https://github.com/{repo}/issues/{selected[0]}")

    def stop_bridge(self):
        self.stop.set()
        self.commands.put(None)
        self.status.set("Stopping; submitted GitHub operations may continue")

    def pump(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "rows":
                    self.rows = data
                    for number, row in data.items():
                        values = (number, row["status"], row.get("detail", ""))
                        if self.jobs.exists(number):
                            self.jobs.item(number, values=values)
                        else:
                            self.jobs.insert("", "end", iid=number, values=values)
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
        self.stop_bridge()
        self.closed = True
        if self.after_id:
            self.window.after_cancel(self.after_id)
        self.window.destroy()
