from acc_desktop_target import DesktopTargetMixin
import json
import base64
import math
import os
import queue
import re
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from acc_task_state import CodexState, CurrentTask, DesktopSnapshot, handoff_mtime

APP_TITLE = "ACC v1.3 - Agent Codex Controller"
CONFIG_FILE = Path.home() / ".acc_agent_codex_controller.json"
HISTORY_FILE_NAME = "ACC_HISTORY.log"
HANDOFF_FILE_NAME = "ACC_HANDOFF.md"

MODE_DESKTOP = "Desktop App"
MODE_CLI = "CLI"

DEFAULT_PROMPT = '''Continue working on the current project from its present repository state.

Do not restart the project from scratch.
1. Check git status and recent changes.
2. Read AGENTS.md, PLANS.md, README files, and project instructions if present.
3. Determine where the previous Codex session stopped.
4. Continue unfinished work without repeating completed work.
5. Preserve fresh/generated data files unless changes are necessary.
6. Run available tests/checks before publishing or committing.
7. Work autonomously until a real blocker is reached.
8. At the end, write a concise handoff summary to ACC_HANDOFF.md in the project root.
'''

LIMIT_MESSAGES = (
    "usage limit",
    "rate limit",
    "limit reached",
    "quota exceeded",
    "exceeded your quota",
    "try again later",
    "you've hit your limit",
    "you have hit your limit",
    "you have reached your usage limit",
    "usage cap",
    "too many requests",
)


@dataclass
class AgentProfile:
    name: str = ""
    url: str = ""
    instructions: str = ""


class ACCApp(DesktopTargetMixin):
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1180x880")
        self.root.minsize(1000, 740)

        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.background = threading.local()
        self.run_config = None
        self.stop_event = threading.Event()
        self.worker = None
        self.github_panel = None
        self.current_process = None
        self.latest_codex_output = ""
        self.last_handoff_text = ""
        self.last_handoff_mtime = 0.0
        self.last_handoff_path = None
        self.session_started_at = None
        self.next_run_at = None
        self.deadline = None
        self.current_task = None
        self.codex_state = CodexState.IDLE
        self.task_lock = threading.RLock()
        self.task_file = CONFIG_FILE.with_suffix(".task.json")
        self.task_load_error = ""
        self.load_current_task()

        self.settings = self.load_settings()
        self.agents = [
            AgentProfile(**{k: x[k] for k in ("name", "url", "instructions") if k in x})
            for x in self.settings.get("agents", [])
            if isinstance(x, dict) and all(isinstance(v, str) for v in x.values())
        ]

        self.build_ui()
        self.apply_settings()
        self.update_mode_ui()
        if self.current_task and self.current_task.pending:
            self.codex_state = CodexState.STOPPED
            self.render_codex_state()
            self.enqueue_log("Unresolved Codex task restored. START resumes monitoring without submitting a prompt.")
        if self.task_load_error:
            self.codex_state = CodexState.ERROR
            self.render_codex_state()
            self.enqueue_log(self.task_load_error)
        self.root.after(200, self.poll_log_queue)
        self.root.after(1000, self.watch_handoff_file)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        project_box = ttk.LabelFrame(top, text="Codex project", padding=8)
        project_box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ttk.Label(project_box, text="Project directory:").grid(
            row=0, column=0, sticky="w"
        )
        self.project_var = tk.StringVar()
        ttk.Entry(project_box, textvariable=self.project_var).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(project_box, text="Browse...", command=self.choose_project).grid(
            row=0, column=2
        )

        ttk.Label(project_box, text="Codex mode:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.mode_var = tk.StringVar(value=MODE_DESKTOP)
        self.mode_combo = ttk.Combobox(
            project_box,
            textvariable=self.mode_var,
            values=(MODE_DESKTOP, MODE_CLI),
            state="readonly",
            width=18,
        )
        self.mode_combo.grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_mode_changed())

        self.test_btn = ttk.Button(project_box, text="Test Codex", command=self.test_codex)
        self.test_btn.grid(row=1, column=2, pady=(6, 0))
        ttk.Button(project_box, text="Resolve delivery / choose window", command=self.reset_desktop_delivery).grid(row=2, column=2, pady=(6, 0))

        ttk.Label(project_box, text="CLI command:").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.codex_var = tk.StringVar(value="codex")
        self.codex_entry = ttk.Entry(project_box, textvariable=self.codex_var)
        self.codex_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=(6, 0))

        ttk.Label(project_box, text="Legacy title (unused):").grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )
        self.desktop_title_var = tk.StringVar(value="Codex")
        self.desktop_title_entry = ttk.Entry(project_box, textvariable=self.desktop_title_var)
        self.desktop_title_entry.grid(row=3, column=1, sticky="ew", padx=6, pady=(6, 0))
        ttk.Label(
            project_box,
            text="Keep the target Codex chat and prompt input selected",
        ).grid(row=3, column=2, sticky="w", pady=(6, 0))

        self.desktop_inspect_var = tk.BooleanVar(value=True)
        self.desktop_inspect_check = ttk.Checkbutton(
            project_box,
            text="Inspect Desktop UI for usage-limit text when pywinauto is available",
            variable=self.desktop_inspect_var,
        )
        self.desktop_inspect_check.grid(
            row=4, column=1, columnspan=2, sticky="w", padx=6, pady=(6, 0)
        )
        project_box.columnconfigure(1, weight=1)

        timing_box = ttk.LabelFrame(top, text="Schedule", padding=8)
        timing_box.pack(side="left", fill="y", padx=(5, 0))

        self.interval_var = tk.StringVar(value="1")
        self.interval_unit_var = tk.StringVar(value="hours")
        self.days_var = tk.StringVar(value="1")
        self.repeat_var = tk.BooleanVar(value=True)
        self.desktop_wait_var = tk.StringVar(value="5")
        self.poll_seconds_var = tk.StringVar(value="30")

        ttk.Label(timing_box, text="Interval:").grid(row=0, column=0, sticky="w")
        ttk.Entry(timing_box, textvariable=self.interval_var, width=7).grid(
            row=0, column=1, padx=4
        )
        ttk.Combobox(
            timing_box,
            textvariable=self.interval_unit_var,
            values=("minutes", "hours"),
            state="readonly",
            width=9,
        ).grid(row=0, column=2)

        ttk.Label(timing_box, text="Days:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(timing_box, textvariable=self.days_var, width=7).grid(
            row=1, column=1, padx=4, pady=(6, 0)
        )
        ttk.Label(timing_box, text="0 = unlimited").grid(
            row=1, column=2, sticky="w", pady=(6, 0)
        )

        ttk.Label(timing_box, text="Inspect after send:").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        self.desktop_wait_entry = ttk.Entry(timing_box, textvariable=self.desktop_wait_var, width=7)
        self.desktop_wait_entry.grid(row=2, column=1, padx=4, pady=(6, 0))
        ttk.Label(timing_box, text="seconds").grid(
            row=2, column=2, sticky="w", pady=(6, 0)
        )

        ttk.Checkbutton(timing_box, text="Repeat", variable=self.repeat_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        ttk.Label(timing_box, text="Monitor every:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(timing_box, textvariable=self.poll_seconds_var, width=7).grid(row=4, column=1)
        ttk.Label(timing_box, text="seconds").grid(row=4, column=2, sticky="w")

        agent_box = ttk.LabelFrame(outer, text="Web ChatGPT agent / project chat", padding=8)
        agent_box.pack(fill="x", pady=(8, 8))
        ttk.Label(agent_box, text="Agent:").grid(row=0, column=0, sticky="w")
        self.agent_var = tk.StringVar()
        self.agent_combo = ttk.Combobox(agent_box, textvariable=self.agent_var, state="readonly")
        self.agent_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.agent_combo.bind("<<ComboboxSelected>>", lambda e: self.save_settings())
        ttk.Button(agent_box, text="Add / Edit", command=self.edit_agent).grid(
            row=0, column=2, padx=(0, 4)
        )
        ttk.Button(agent_box, text="Remove", command=self.remove_agent).grid(
            row=0, column=3, padx=(0, 4)
        )
        ttk.Button(agent_box, text="Open ChatGPT", command=self.open_agent).grid(
            row=0, column=4
        )
        agent_box.columnconfigure(1, weight=1)

        middle = ttk.Panedwindow(outer, orient="horizontal")
        middle.pack(fill="both", expand=True, pady=(0, 8))
        left = ttk.Frame(middle)
        right = ttk.Frame(middle)
        middle.add(left, weight=1)
        middle.add(right, weight=1)

        prompt_box = ttk.LabelFrame(left, text="Prompt sent to Codex", padding=8)
        prompt_box.pack(fill="both", expand=True)
        self.prompt_text = tk.Text(prompt_box, height=12, wrap="word")
        self.prompt_text.pack(fill="both", expand=True)

        notes_box = ttk.LabelFrame(left, text="ChatGPT -> Codex notes", padding=8)
        notes_box.pack(fill="both", expand=True, pady=(8, 0))
        self.agent_notes = tk.Text(notes_box, height=10, wrap="word")
        self.agent_notes.pack(fill="both", expand=True)
        note_buttons = ttk.Frame(notes_box)
        note_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(note_buttons, text="Send to Codex", command=self.send_notes_to_codex).pack(side="left")
        ttk.Button(
            note_buttons,
            text="Clear notes",
            command=lambda: self.agent_notes.delete("1.0", "end"),
        ).pack(side="left", padx=(6, 0))

        handoff_box = ttk.LabelFrame(right, text="Codex -> ChatGPT handoff", padding=8)
        handoff_box.pack(fill="both", expand=True)
        self.handoff_text = tk.Text(handoff_box, height=20, wrap="word", state="disabled")
        self.handoff_text.pack(fill="both", expand=True)
        handoff_buttons = ttk.Frame(handoff_box)
        handoff_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(handoff_buttons, text="Send to ChatGPT", command=self.send_to_chatgpt).pack(side="left")
        ttk.Button(handoff_buttons, text="Copy handoff", command=self.copy_handoff).pack(side="left", padx=(6, 0))
        ttk.Button(handoff_buttons, text="Open file", command=self.open_handoff).pack(side="left", padx=(6, 0))
        ttk.Button(handoff_buttons, text="Reload", command=self.load_handoff_now).pack(side="left", padx=(6, 0))
        self.handoff_status_var = tk.StringVar(value="No handoff yet")
        ttk.Label(handoff_box, textvariable=self.handoff_status_var).pack(anchor="w", pady=(6, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        self.start_btn = ttk.Button(controls, text="START", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="STOP", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Save settings", command=self.save_settings).pack(
            side="left", padx=(16, 0)
        )
        ttk.Button(controls, text="GitHub / Mobile", command=self.open_github_panel).pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")
        self.codex_status_var = tk.StringVar(value="Codex status: Idle")
        ttk.Label(controls, textvariable=self.codex_status_var, font=("Segoe UI", 10, "bold")).pack(side="right", padx=16)

        log_box = ttk.LabelFrame(outer, text="ACC log", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_box, height=10, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.refresh_agent_combo()

    def load_settings(self):
        if not CONFIG_FILE.exists():
            return {}
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            if not isinstance(data.get("agents", []), list):
                data["agents"] = []
            return data
        except Exception:
            return {}

    def save_settings(self):
        data = {
            "project": self.project_var.get().strip(),
            "codex_mode": self.mode_var.get(),
            "codex_command": self.codex_var.get().strip(),
            "desktop_window_title": self.desktop_title_var.get().strip(),
            "desktop_inspect": bool(self.desktop_inspect_var.get()),
            "desktop_wait": self.desktop_wait_var.get().strip(),
            "poll_seconds": self.poll_seconds_var.get().strip(),
            "interval": self.interval_var.get().strip(),
            "interval_unit": self.interval_unit_var.get(),
            "days": self.days_var.get().strip(),
            "repeat": bool(self.repeat_var.get()),
            "selected_agent": self.agent_var.get(),
            "agent_notes": self.agent_notes.get("1.0", "end-1c"),
            "prompt": self.prompt_text.get("1.0", "end-1c"),
            "agents": [asdict(a) for a in self.agents],
        }
        try:
            CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.enqueue_log("Settings saved.")
        except Exception as exc:
            self.enqueue_log(f"Could not save settings: {exc}")

    def apply_settings(self):
        self.project_var.set(self.settings.get("project", str(Path(__file__).resolve().parent)))
        self.mode_var.set(self.settings.get("codex_mode", MODE_DESKTOP))
        self.codex_var.set(self.settings.get("codex_command", "codex"))
        self.desktop_title_var.set(self.settings.get("desktop_window_title", "Codex"))
        self.desktop_inspect_var.set(bool(self.settings.get("desktop_inspect", True)))
        self.desktop_wait_var.set(self.settings.get("desktop_wait", "5"))
        self.poll_seconds_var.set(self.settings.get("poll_seconds", "30"))
        self.interval_var.set(self.settings.get("interval", "1"))
        self.interval_unit_var.set(self.settings.get("interval_unit", "hours"))
        self.days_var.set(self.settings.get("days", "1"))
        self.repeat_var.set(bool(self.settings.get("repeat", True)))
        self.prompt_text.insert("1.0", self.settings.get("prompt", DEFAULT_PROMPT))
        self.agent_notes.insert("1.0", self.settings.get("agent_notes", ""))

        selected = self.settings.get("selected_agent", "")
        names = [a.name for a in self.agents]
        if selected in names:
            self.agent_var.set(selected)
        elif names:
            self.agent_var.set(names[0])

    def choose_project(self):
        folder = filedialog.askdirectory(initialdir=self.project_var.get() or None)
        if folder:
            self.project_var.set(folder)
            self.save_settings()
            self.load_handoff_now()

    def refresh_agent_combo(self):
        names = [a.name for a in self.agents]
        self.agent_combo["values"] = names
        if self.agent_var.get() not in names:
            self.agent_var.set(names[0] if names else "")

    def selected_agent(self):
        return next((a for a in self.agents if a.name == self.agent_var.get()), None)

    def edit_agent(self):
        current = self.selected_agent()
        name = simpledialog.askstring(
            "ACC - Agent",
            "Agent / ChatGPT project name:",
            initialvalue=current.name if current else "",
            parent=self.root,
        )
        if not name:
            return
        url = simpledialog.askstring(
            "ACC - Agent",
            "URL of the ChatGPT conversation/project:",
            initialvalue=current.url if current else "",
            parent=self.root,
        )
        if url is None:
            return
        instructions = simpledialog.askstring(
            "ACC - Agent",
            "Optional collaboration instructions:",
            initialvalue=current.instructions if current else "",
            parent=self.root,
        )
        if instructions is None:
            return

        profile = AgentProfile(name.strip(), url.strip(), instructions.strip())
        if current:
            self.agents[self.agents.index(current)] = profile
        else:
            existing = next((a for a in self.agents if a.name == profile.name), None)
            if existing:
                self.agents[self.agents.index(existing)] = profile
            else:
                self.agents.append(profile)

        self.refresh_agent_combo()
        self.agent_var.set(profile.name)
        self.save_settings()

    def remove_agent(self):
        agent = self.selected_agent()
        if not agent:
            return
        if messagebox.askyesno("ACC", f"Remove agent profile '{agent.name}'?"):
            self.agents.remove(agent)
            self.refresh_agent_combo()
            self.save_settings()

    def open_agent(self):
        agent = self.selected_agent()
        if not agent or not agent.url:
            messagebox.showinfo("ACC", "Add a URL to the selected ChatGPT agent/project first.")
            return False
        webbrowser.open(agent.url)
        return True

    def on_mode_changed(self):
        self.update_mode_ui()
        self.save_settings()

    def update_mode_ui(self):
        mode = self.mode_var.get()
        if mode == MODE_DESKTOP:
            self.codex_entry.configure(state="disabled")
            self.desktop_title_entry.configure(state="disabled")
            self.desktop_inspect_check.configure(state="normal")
            self.desktop_wait_entry.configure(state="normal")
        else:
            self.codex_entry.configure(state="normal")
            self.desktop_title_entry.configure(state="disabled")
            self.desktop_inspect_check.configure(state="disabled")
            self.desktop_wait_entry.configure(state="disabled")

    def validate_inputs(self):
        project_text = self.get_value("project_var").strip()
        if not project_text:
            raise ValueError("Project directory cannot be empty.")
        project = Path(project_text)
        if not project.exists() or not project.is_dir():
            raise ValueError("Project directory does not exist.")

        try:
            interval = float(self.get_value("interval_var"))
            days = float(self.get_value("days_var"))
            desktop_wait = float(self.get_value("desktop_wait_var"))
            poll_seconds = float(self.get_value("poll_seconds_var"))
        except ValueError:
            raise ValueError("Interval, Days, Inspect delay and Monitor interval must be numbers.")

        if not all(math.isfinite(value) for value in (interval, days, desktop_wait, poll_seconds)):
            raise ValueError("Schedule values must be finite numbers.")
        if self.get_value("mode_var") not in (MODE_DESKTOP, MODE_CLI):
            raise ValueError("Select a valid Codex mode.")
        if self.get_value("interval_unit_var") not in ("minutes", "hours"):
            raise ValueError("Select minutes or hours for the interval.")
        if interval <= 0:
            raise ValueError("Interval must be greater than zero.")
        if days < 0:
            raise ValueError("Days cannot be negative.")
        if desktop_wait < 0:
            raise ValueError("Desktop inspect delay cannot be negative.")
        if not 1 <= poll_seconds <= 3600:
            raise ValueError("Monitor interval must be between 1 and 3600 seconds.")

        try:
            datetime.now() + timedelta(days=days)
            datetime.now() + timedelta(seconds=self.interval_seconds())
            if desktop_wait > threading.TIMEOUT_MAX:
                raise OverflowError
        except OverflowError:
            raise ValueError("Schedule values are too large.")

        if self.get_value("mode_var") == MODE_DESKTOP:
            if os.name != "nt":
                raise ValueError("Desktop App mode currently supports Windows only.")

        return project, interval, days, desktop_wait

    def interval_seconds(self):
        value = float(self.get_value("interval_var"))
        return value * (60 if self.get_value("interval_unit_var") == "minutes" else 3600)

    def get_value(self, name):
        settings = getattr(self.background, "settings", None)
        return settings[name] if settings is not None else getattr(self, name).get()

    def capture_config(self):
        names = ("project_var", "mode_var", "codex_var", "desktop_title_var",
                 "desktop_inspect_var", "desktop_wait_var", "interval_var",
                 "interval_unit_var", "days_var", "repeat_var", "poll_seconds_var")
        return {**{name: getattr(self, name).get() for name in names},
                "prompt": self.build_codex_prompt()}

    def run_background(self, target, config):
        self.background.settings = config
        try:
            target()
        finally:
            del self.background.settings

    def build_codex_prompt(self):
        settings = getattr(self.background, "settings", None)
        if settings is not None:
            return settings["prompt"]
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        agent = self.selected_agent()
        notes = self.agent_notes.get("1.0", "end-1c").strip()
        extra = []

        if agent:
            extra += ["", "ACC collaboration context:", f"- Selected Web ChatGPT agent/project: {agent.name}"]
            if agent.url:
                extra.append(f"- Chat URL: {agent.url}")
            if agent.instructions:
                extra.append(f"- Collaboration instructions: {agent.instructions}")
            extra.append(f"- Use {HANDOFF_FILE_NAME} as the local Codex-to-ChatGPT handoff channel.")

        if notes:
            extra += ["", "Latest instructions/notes from Web ChatGPT:", notes]

        return prompt + "\n" + "\n".join(extra)

    def test_codex(self):
        target = self.test_desktop_codex if self.mode_var.get() == MODE_DESKTOP else self.test_cli_codex
        threading.Thread(target=self.run_background, args=(target, self.capture_config()), daemon=True).start()

    def test_cli_codex(self):
        command = self.get_value("codex_var").strip() or "codex"
        try:
            result = subprocess.run(
                [command, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            out = ((result.stdout or "") + (result.stderr or "")).strip()
            self.enqueue_log(f"CLI test returned {result.returncode}: {out or '(no output)'}")
        except FileNotFoundError:
            self.enqueue_log("CLI test failed: Codex command was not found.")
        except Exception as exc:
            self.enqueue_log(f"CLI test failed: {exc}")



    def start(self):
        if self.github_panel and self.github_panel.thread and self.github_panel.thread.is_alive():
            messagebox.showerror("ACC", "Stop the GitHub publishing bridge before starting the local Codex schedule.")
            return
        if self.worker and self.worker.is_alive():
            return

        try:
            self.validate_inputs()
            if self.task_load_error:
                raise ValueError(self.task_load_error)
            if self.current_task and self.current_task.pending:
                if (self.mode_var.get() != MODE_DESKTOP or
                        Path(self.project_var.get().strip()).resolve() != Path(self.current_task.project).resolve()):
                    raise ValueError("An unresolved Desktop task exists. Select its original project and Desktop mode to resume monitoring.")
        except Exception as exc:
            messagebox.showerror("ACC", str(exc))
            return

        self.save_settings()
        self.run_config = self.capture_config()
        self.stop_event.clear()
        if self.current_task is None:
            self.set_task_state(CodexState.IDLE)
        self.session_started_at = datetime.now()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running")
        self.worker = threading.Thread(target=self.run_background,
                                       args=(self.controller_loop, self.run_config), daemon=True)
        self.worker.start()

    def stop(self):
        self.enqueue_log("STOP requested by user.")
        self.stop_event.set()
        self.status_var.set("Stopped")
        self.set_task_state(CodexState.STOPPED)
        self.codex_status_var.set("Codex status: Stopped")

        if self.run_config and self.run_config["mode_var"] == MODE_CLI:
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                    self.enqueue_log("Current Codex CLI process was asked to terminate.")
                except Exception:
                    pass
        else:
            self.enqueue_log(
                "Desktop mode: ACC automation is stopping. "
                "An already-running Codex task is not force-terminated."
            )

    def controller_loop(self):
        try:
            project, _, days, _ = self.validate_inputs()
            deadline = self.session_started_at + timedelta(days=days) if days > 0 else None
            self.deadline = deadline

            self.enqueue_log(f"{APP_TITLE} started. Mode: {self.get_value('mode_var')}. Project: {project}")
            if deadline:
                self.enqueue_log(f"Automatic stop deadline: {deadline:%Y-%m-%d %H:%M:%S}")

            task = self.current_task
            if (task and not task.pending and self.get_value("mode_var") == MODE_DESKTOP
                    and Path(task.project).resolve() == project.resolve()):
                resume_at = task.next_attempt_time
                if resume_at is None and task.parsed_reset_time:
                    resume_at = task.parsed_reset_time + timedelta(seconds=20)
                if resume_at and resume_at > datetime.now():
                    self.next_run_at = resume_at
                    self.set_task_state(CodexState.WAITING if task.parsed_reset_time else CodexState.COMPLETED)
                    self.enqueue_log(f"Resuming scheduled wait until {resume_at:%Y-%m-%d %H:%M:%S}")
                    if self.monitor_wait((resume_at - datetime.now()).total_seconds()):
                        return

            while not self.stop_event.is_set():
                if deadline and datetime.now() >= deadline:
                    self.enqueue_log("Configured working period ended.")
                    break

                status, output, reset_dt = self.run_codex_once(project)
                self.ui_queue.put(lambda p=project, text=output: self.show_run_output(p, text))

                if status == "fatal":
                    self.set_task_state(CodexState.ERROR)
                if self.stop_event.is_set() or status in ("fatal", "stopped"):
                    break
                if status == "success":
                    self.set_task_state(CodexState.COMPLETED, "Codex task completed")
                    self.append_history("CODEX TASK COMPLETED", output)
                elif status == "limit":
                    self.set_task_state(CodexState.WAITING, "Usage limit detected")
                else:
                    self.set_task_state(CodexState.ERROR, "Codex task failed")
                if not self.get_value("repeat_var"):
                    break

                wait_seconds = self.interval_seconds()

                if status == "limit" and reset_dt:
                    wait_seconds = max(1, (reset_dt - datetime.now()).total_seconds() + 20)
                    self.enqueue_log(
                        f"Codex reset parsed. Next attempt near: {reset_dt:%Y-%m-%d %H:%M:%S}"
                    )
                elif status == "limit":
                    self.enqueue_log(
                        "Codex limit is active; exact reset time was not parsed, "
                        "so the normal interval is used."
                    )

                if deadline:
                    wait_seconds = min(
                        wait_seconds,
                        max(0, (deadline - datetime.now()).total_seconds()),
                    )

                self.next_run_at = datetime.now() + timedelta(seconds=wait_seconds)
                if self.current_task and not self.current_task.pending and self.get_value("mode_var") == MODE_DESKTOP:
                    self.current_task.next_attempt_time = self.next_run_at
                    self.persist_task()
                label = "Waiting for reset until" if status == "limit" else "Next cycle scheduled for"
                self.enqueue_log(f"{label} {self.next_run_at:%Y-%m-%d %H:%M:%S}")

                if self.stop_event.wait(wait_seconds):
                    break

        except Exception as exc:
            self.set_task_state(CodexState.ERROR)
            self.enqueue_log(f"Controller error: {exc}")
        finally:
            self.current_process = None
            self.next_run_at = None
            if self.stop_event.is_set() or (self.deadline and datetime.now() >= self.deadline):
                self.set_task_state(CodexState.STOPPED)
            self.enqueue_log("ACC stopped.")
            self.ui_queue.put(self.set_stopped_ui)

    def run_codex_once(self, project):
        if self.get_value("mode_var") == MODE_DESKTOP:
            return self.run_desktop_codex_once(project)
        return self.run_cli_codex_once(project)

    def run_cli_codex_once(self, project):
        if self.stop_event.is_set():
            return "stopped", "", None
        self.set_task_state(CodexState.SENDING, "Codex task started")
        command = self.get_value("codex_var").strip() or "codex"
        prompt = self.build_codex_prompt()
        self.append_history("ACC -> CODEX CLI", prompt)
        self.enqueue_log("Starting Codex CLI...")

        try:
            proc = subprocess.Popen(
                [command, "exec", "-"],
                cwd=str(project),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.current_process = proc
            self.set_task_state(CodexState.WORKING)
        except FileNotFoundError:
            self.enqueue_log(
                "FATAL: Codex CLI command not found. "
                "Switch to Desktop App mode or configure the CLI command."
            )
            return "fatal", "", None
        except Exception as exc:
            self.enqueue_log(f"Could not start Codex CLI: {exc}")
            return "retry", str(exc), None

        # communicate drains stdout while writing stdin, including for large prompts.
        output = ""
        try:
            first = True
            while True:
                try:
                    output, _ = proc.communicate(input=prompt if first else None, timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    first = False
                    if self.monitor_stopped():
                        proc.terminate()
                        try:
                            output, _ = proc.communicate(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            output, _ = proc.communicate()
                        break
            return_code = proc.returncode
        finally:
            self.current_process = None

        output = output or ""
        for line in output.splitlines():
            self.enqueue_log(f"CODEX | {line}")
        self.append_history("CODEX CLI OUTPUT", output)
        reset_dt = self.parse_reset_time(output)
        if self.monitor_stopped():
            return "stopped", output, None

        if return_code == 0:
            self.enqueue_log("Codex CLI completed successfully.")
            return "success", output, reset_dt

        if self.looks_like_limit(output):
            return "limit", output, reset_dt

        self.enqueue_log(f"Codex CLI exited with code {return_code}.")
        return "retry", output, reset_dt

    def load_current_task(self):
        if not self.task_file.exists():
            return
        try:
            self.current_task = CurrentTask.from_dict(json.loads(self.task_file.read_text(encoding="utf-8")))
        except Exception as exc:
            self.task_load_error = f"Cannot read pending-task journal {self.task_file}: {exc}. Restore the journal before starting; task status is unknown."

    def persist_task(self):
        with self.task_lock:
            if self.current_task is None:
                return
            temporary = self.task_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.current_task.to_dict(), indent=2), encoding="utf-8")
            temporary.replace(self.task_file)

    def set_task_state(self, state, message=None):
        with self.task_lock:
            if self.stop_event.is_set():
                state = CodexState.STOPPED
            changed = state != self.codex_state
            self.codex_state = state
            if self.current_task:
                self.current_task.state = state
        self.ui_queue.put(self.render_codex_state)
        if state == CodexState.COMPLETED and self.current_task and hasattr(self, '_desktop_target'):
            self._desktop_target.completed(str(self.current_task.task_start_ns))
        if changed and message:
            self.enqueue_log(message)

    def render_codex_state(self):
        with self.task_lock:
            self.codex_status_var.set(f"Codex status: {self.codex_state.value}")

    def monitor_stopped(self):
        return self.stop_event.is_set() or (self.deadline is not None and datetime.now() >= self.deadline)

    def monitor_wait(self, seconds):
        if self.deadline:
            seconds = min(seconds, max(0, (self.deadline - datetime.now()).total_seconds()))
        self.stop_event.wait(seconds)
        return self.monitor_stopped()

    def run_desktop_codex_once(self, project):
        if self.monitor_stopped():
            return "stopped", "", None
        if self.current_task and self.current_task.pending:
            self.enqueue_log("Resuming monitoring of the unresolved Codex task; no prompt submitted.")
            return self.monitor_desktop_task(project)

        ok, info = self.find_desktop_codex_window()
        if not ok:
            self.set_task_state(CodexState.WAITING if info == "USAGE_LIMIT" else CodexState.ERROR, info)
            return ("limit" if info == "USAGE_LIMIT" else "retry"), info, None
        _, pid, handle, _ = info.split("|", 3)
        task = CurrentTask(project=str(project.resolve()),
                           handoff_mtime_before=handoff_mtime(project / HANDOFF_FILE_NAME),
                           window_pid=int(pid), window_handle=int(handle))
        self.current_task = task
        before = self.inspect_desktop_task(task)
        task.baseline_messages = before.messages
        if before.working:
            task.observed_working = True
            self.persist_task()
            self.set_task_state(CodexState.WORKING, "Codex is already working; monitoring without sending a duplicate")
            return self.monitor_desktop_task(project)

        # Persist before sending: a timeout or process crash can leave delivery ambiguous.
        self.persist_task()
        if self.monitor_stopped():
            return "stopped", "", None
        self.set_task_state(CodexState.SENDING, "Codex task started")
        prompt = self.build_codex_prompt()
        self.enqueue_log("Desktop submission prepared; message content omitted from diagnostics.")
        ok, info = self.activate_and_send_to_desktop(prompt)
        if ok:
            task.prompt_sent_time = datetime.now()
            self.enqueue_log("Prompt was pasted and submitted to Codex Desktop; monitoring completion.")
        else:
            self.enqueue_log(f"Desktop delivery could not be confirmed: {info}. No automatic resubmission.")
        self.persist_task()
        if self.monitor_stopped():
            return "stopped", "", None
        self.set_task_state(CodexState.WORKING)
        initial_delay = float(self.get_value("desktop_wait_var"))
        if ok and initial_delay > 0 and not task.new_handoff(handoff_mtime(project / HANDOFF_FILE_NAME)):
            self.monitor_wait(initial_delay)
        return self.monitor_desktop_task(project)

    def monitor_desktop_task(self, project):
        task = self.current_task
        self.set_task_state(CodexState.WORKING)
        last_detail = None
        last_report = time.monotonic()
        while not self.monitor_stopped():
            mtime = handoff_mtime(project / HANDOFF_FILE_NAME)
            # Read the strongest signal first, avoiding a slow UIA call when it exists.
            snapshot = DesktopSnapshot() if task.new_handoff(mtime) else self.inspect_desktop_task(task)
            if self.monitor_stopped():
                break
            state, detail = task.observe(snapshot, mtime)
            if snapshot.detail != last_detail:
                if snapshot.detail:
                    self.enqueue_log(snapshot.detail)
                last_detail = snapshot.detail
            if state == CodexState.COMPLETED:
                task.pending = False
                task.next_attempt_time = datetime.now() + timedelta(seconds=self.interval_seconds())
                self.set_task_state(state, detail)
                self.persist_task()
                self.ui_queue.put(lambda p=project: self.show_run_output(p, ""))
                self.append_history("DESKTOP TASK COMPLETED", detail)
                return "success", detail, None
            if state == CodexState.WAITING:
                task.pending = False
                task.parsed_reset_time = self.parse_reset_time(detail)
                task.next_attempt_time = (task.parsed_reset_time + timedelta(seconds=20)
                                          if task.parsed_reset_time else
                                          datetime.now() + timedelta(seconds=self.interval_seconds()))
                self.set_task_state(state, "Usage limit detected")
                self.persist_task()
                self.append_history("CODEX USAGE LIMIT", detail)
                return "limit", detail, task.parsed_reset_time
            if state == CodexState.ERROR:
                # Failure text alone does not prove that the old task cannot resume.
                self.set_task_state(state, detail)
                self.persist_task()
                return "fatal", detail, None
            if time.monotonic() - last_report >= 300:
                self.enqueue_log("Codex is still working" if snapshot.working else "Awaiting task completion evidence; no duplicate prompt will be sent.")
                last_report = time.monotonic()
            self.persist_task()
            if self.monitor_wait(float(self.get_value("poll_seconds_var"))):
                break
        self.set_task_state(CodexState.STOPPED)
        self.persist_task()
        return "stopped", "", None



    def inspect_desktop_task(self, task):
        if not self.get_value("desktop_inspect_var"):
            return DesktopSnapshot(detail="Desktop UI inspection disabled; monitoring handoff only. No timed resubmission.")
        if task.window_pid is None or task.window_handle is None:
            return DesktopSnapshot(detail="No validated Codex window; monitoring handoff only.")
        import sys
        try:
            request = json.dumps({"pid": task.window_pid, "handle": task.window_handle})
            result = self.run_cancellable_process(
                [sys.executable, str(Path(__file__).with_name("acc_desktop_probe.py"))],
                request, timeout=10)
            if result.returncode:
                return DesktopSnapshot(detail="Codex UI probe unavailable; monitoring handoff only.")
            data = json.loads(result.stdout)
            data["messages"] = tuple(data.get("messages", ()))
            return DesktopSnapshot(**data)
        except Exception:
            return DesktopSnapshot(detail="Codex UI probe unavailable; monitoring handoff only.")

    def run_cancellable_process(self, command, text, timeout, env=None):
        if self.monitor_stopped():
            raise RuntimeError("ACC stopped")
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        end = time.monotonic() + timeout
        first = True
        try:
            while not self.monitor_stopped():
                try:
                    out, err = proc.communicate(input=text if first else None, timeout=0.1)
                    return subprocess.CompletedProcess(command, proc.returncode, out, err)
                except subprocess.TimeoutExpired:
                    first = False
                    if time.monotonic() >= end:
                        raise TimeoutError("Desktop helper timed out; delivery/state may be unknown")
            raise RuntimeError("ACC stopped")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()

    @staticmethod
    def looks_like_limit(output):
        text = (output or "").lower()
        return any(token in text for token in LIMIT_MESSAGES)

    @staticmethod
    def parse_reset_time(output):
        if not output:
            return None

        now = datetime.now()

        relative = re.search(
            r"(?:resets?|try again|available)\s+in\s+"
            r"((?:\d+\s*(?:hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b[\s,]*(?:and\s*)?)+)",
            output, re.I,
        )
        if relative:
            seconds = sum(int(value) * {"h": 3600, "m": 60, "s": 1}[unit.lower()[0]]
                          for value, unit in re.findall(r"(\d+)\s*([a-z]+)", relative.group(1), re.I))
            try:
                return now + timedelta(seconds=seconds)
            except OverflowError:
                return None

        tm = re.search(
            r"(?:reset(?:s)?|try again|available)\s+(?:at|after)\s+(\d{1,2}):(\d{2})\s*(am|pm)?",
            output,
            re.I,
        )
        if tm:
            hour = int(tm.group(1))
            minute = int(tm.group(2))
            ampm = tm.group(3)

            if ampm:
                if not 1 <= hour <= 12:
                    return None
                if hour == 12:
                    hour = 0
                if ampm.lower() == "pm":
                    hour += 12

            if 0 <= hour <= 23 and 0 <= minute <= 59:
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(days=1)
                return candidate

        iso = re.search(
            r"(20\d{2}-\d{2}-\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?",
            output,
        )
        if iso:
            try:
                sec = int(iso.group(4) or 0)
                return datetime.strptime(
                    f"{iso.group(1)} {iso.group(2)}:{iso.group(3)}:{sec:02d}",
                    "%Y-%m-%d %H:%M:%S",
                )
            except ValueError:
                pass

        return None

    def handoff_path(self):
        return Path(self.project_var.get().strip()) / HANDOFF_FILE_NAME

    def show_run_output(self, project, output):
        if Path(self.project_var.get().strip()) == project:
            self.load_handoff_now()
            self.latest_codex_output = output

    def history_path(self):
        return Path(self.get_value("project_var").strip()) / HISTORY_FILE_NAME

    def append_history(self, kind, text):
        try:
            path = self.history_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(
                    f"\n\n===== {datetime.now():%Y-%m-%d %H:%M:%S} | {kind} =====\n"
                )
                file.write(text or "")
                file.write("\n")
        except Exception:
            pass

    def load_handoff_now(self):
        path = self.handoff_path()
        if path != self.last_handoff_path:
            self.last_handoff_path = path
            self.latest_codex_output = ""
            self.last_handoff_mtime = 0.0
            self.last_handoff_text = ""
        if not path.exists():
            self.last_handoff_text = ""
            self.last_handoff_mtime = 0.0
            self.handoff_text.configure(state="normal")
            self.handoff_text.delete("1.0", "end")
            self.handoff_text.configure(state="disabled")
            self.handoff_status_var.set("No handoff yet")
            return

        try:
            text = path.read_text(encoding="utf-8")
            mtime = path.stat().st_mtime
        except Exception as exc:
            self.enqueue_log(f"Could not read handoff: {exc}")
            return

        is_new = (
            mtime > self.last_handoff_mtime
            and text.strip()
            and text != self.last_handoff_text
        )
        self.last_handoff_mtime = mtime
        self.last_handoff_text = text

        self.handoff_text.configure(state="normal")
        self.handoff_text.delete("1.0", "end")
        self.handoff_text.insert("1.0", text)
        self.handoff_text.configure(state="disabled")

        if text.strip():
            self.handoff_status_var.set(
                f"Handoff loaded: {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S}"
            )
        else:
            self.handoff_status_var.set("Handoff file is empty")

        if is_new:
            self.enqueue_log("New Codex handoff detected.")
            self.append_history("CODEX -> CHATGPT HANDOFF", text)
            self.root.bell()

    def watch_handoff_file(self):
        try:
            self.load_handoff_now()
        finally:
            self.root.after(1500, self.watch_handoff_file)

    def copy_handoff(self):
        text = self.last_handoff_text.strip() or self.latest_codex_output.strip()
        if not text:
            messagebox.showinfo("ACC", "No handoff/output available yet.")
            return False

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.enqueue_log("Handoff copied to clipboard.")
        return True

    def send_to_chatgpt(self):
        if not self.copy_handoff():
            return

        opened = self.open_agent()
        if opened:
            self.enqueue_log(
                "Selected ChatGPT chat opened. Handoff is in the clipboard; paste and send it there."
            )
            self.handoff_status_var.set("Copied + ChatGPT opened")

    def send_notes_to_codex(self):
        notes = self.agent_notes.get("1.0", "end-1c").strip()
        if not notes:
            messagebox.showinfo("ACC", "Enter ChatGPT instructions/notes first.")
            return

        self.save_settings()
        self.append_history("CHATGPT -> CODEX NOTES", notes)
        if self.run_config is not None:
            self.run_config["prompt"] = self.build_codex_prompt()
        self.enqueue_log("ChatGPT notes saved for the next Codex run.")

        if not (self.worker and self.worker.is_alive()):
            if messagebox.askyesno("ACC", "Send these notes to Codex now?"):
                self.start()

    def open_handoff(self):
        path = self.handoff_path()
        if not path.exists():
            messagebox.showinfo("ACC", f"Handoff file does not exist yet:\n{path}")
            return

        try:
            os.startfile(str(path))
        except Exception as exc:
            messagebox.showerror("ACC", str(exc))

    def enqueue_log(self, message):
        self.log_queue.put((f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}",
                            self.get_value("project_var")))

    def poll_log_queue(self):
        try:
            while True:
                self.append_log(*self.log_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(150, self.poll_log_queue)

    def append_log(self, line, project_text):
        project_text = project_text.strip()
        project = Path(project_text) if project_text else Path.cwd()
        log_dir = project if project.exists() and project.is_dir() else Path.cwd()

        try:
            with (log_dir / "acc.log").open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except Exception:
            pass

        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_stopped_ui(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopped")

    def on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("ACC", "ACC is running. Stop it and exit?"):
                return

            self.stop_event.set()

            if (
                self.current_process
                and self.current_process.poll() is None
            ):
                try:
                    self.current_process.terminate()
                except Exception:
                    pass

        self.save_settings()
        if self.github_panel and not self.github_panel.closed:
            self.github_panel.close()
        self.root.destroy()

    def open_github_panel(self):
        from acc_github_panel import GitHubPanel
        if self.github_panel and not self.github_panel.closed:
            self.github_panel.window.lift()
            return
        if self.github_panel and self.github_panel.thread and self.github_panel.thread.is_alive():
            messagebox.showinfo("ACC", "The previous bridge is stopping. Wait for its current request to finish.")
            return
        self.github_panel = GitHubPanel(self, CONFIG_FILE.with_suffix(".bridge.json"))


def main():
    mutex = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        mutex = kernel32.CreateMutexW(None, False, "Local\\ACC_v1_3_controller")
        if not mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            kernel32.CloseHandle(mutex)
            messagebox.showinfo("ACC", "ACC v1.3 is already open. Use the existing controller window.")
            return
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    try:
        ACCApp(root)
        root.mainloop()
    finally:
        if mutex:
            kernel32.CloseHandle(mutex)


if __name__ == "__main__":
    main()
