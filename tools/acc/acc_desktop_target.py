"""Fail-closed, window-scoped Codex Desktop delivery. Never use global keystrokes."""

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import threading
import time
from dataclasses import dataclass


class TargetError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Window:
    hwnd: int
    pid: int
    exe: str
    title: str
    created: str


def is_codex(window):
    path = PureWindowsPath(window.exe)
    parts = [part.casefold() for part in path.parts]
    store_root = PureWindowsPath(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'WindowsApps'
    local_root = PureWindowsPath(os.environ.get('LOCALAPPDATA', r'C:\Users\Default\AppData\Local')) / 'OpenAI' / 'Codex'
    packaged = (str(path).casefold().startswith(str(store_root).casefold() + '\\') and any(part.startswith('openai.codex_')
                 and part.endswith('__2p2nqsd0c76g0') for part in parts))
    installed = str(path.parent).casefold() in {str(local_root).casefold(), str(local_root / 'app').casefold()}
    return (window.pid != os.getpid() and path.name.casefold() in {'chatgpt.exe', 'codex.exe'}
            and (packaged or installed) and not window.title.casefold().startswith('acc v'))


class WinBackend:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        import win32gui
        import win32process
        self.gui, self.process = win32gui, win32process
        self.ctypes, self.types = ctypes, wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                         wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4

    def begin(self):
        import pythoncom
        pythoncom.CoInitializeEx(0)

    def end(self):
        import pythoncom
        pythoncom.CoUninitialize()

    def describe(self, hwnd):
        if not self.gui.IsWindow(hwnd):
            raise TargetError('WINDOW_NOT_FOUND')
        _, pid = self.process.GetWindowThreadProcessId(hwnd)
        handle = self.kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            raise TargetError('WINDOW_NOT_FOUND')
        try:
            path = self.ctypes.create_unicode_buffer(32768)
            length = self.types.DWORD(len(path))
            times = [self.types.FILETIME() for _ in range(4)]
            if not self.kernel.QueryFullProcessImageNameW(handle, 0, path, self.ctypes.byref(length)):
                raise TargetError('WINDOW_NOT_FOUND')
            if not self.kernel.GetProcessTimes(handle, *(self.ctypes.byref(t) for t in times)):
                raise TargetError('WINDOW_NOT_FOUND')
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            return Window(hwnd, pid, path.value, self.gui.GetWindowText(hwnd), str(created))
        finally:
            self.kernel.CloseHandle(handle)

    def windows(self):
        result = []
        def visit(hwnd, _):
            try:
                # Only unowned top-level application windows, never child/dialog windows.
                if self.gui.IsWindowVisible(hwnd) and not self.gui.GetWindow(hwnd, 4):
                    window = self.describe(hwnd)
                    if is_codex(window):
                        result.append(window)
            except Exception:
                pass
        self.gui.EnumWindows(visit, None)
        return result

    def foreground(self):
        return self.gui.GetForegroundWindow()

    def activate(self, window):
        try:
            if self.gui.IsIconic(window.hwnd):
                self.gui.ShowWindow(window.hwnd, 9)
            self.gui.SetForegroundWindow(window.hwnd)
            for _ in range(10):
                if self.foreground() == window.hwnd:
                    return True
                time.sleep(0.05)
        except Exception:
            pass
        return False

    def controls(self, window):
        try:
            from pywinauto import Desktop
            root = Desktop(backend='uia').window(handle=window.hwnd).wrapper_object()
            if root.process_id() != window.pid:
                raise TargetError('WINDOW_NOT_FOUND')
            children = root.descendants()
            visible = [c for c in children if c.is_visible()]
            for c in visible:
                if c.element_info.control_type == 'StatusBar' and any(
                        text in c.window_text().casefold() for text in
                        ('usage limit', 'rate limit', 'лимит использования')):
                    raise TargetError('USAGE_LIMIT')
            stops = {'stop', 'stop generating', 'stop task', 'interrupt', 'остановить'}
            sends = {'send', 'send message', 'send prompt', 'submit', 'отправить', 'отправить сообщение'}
            for c in visible:
                if c.element_info.control_type == 'Button' and c.is_enabled() and c.window_text().strip().casefold() in stops:
                    raise TargetError('INPUT_UNAVAILABLE')
            # The user must focus the composer. Never guess by screen position or tab order.
            edits = [c for c in visible if c.element_info.control_type == 'Edit'
                     and c.is_enabled() and c.has_keyboard_focus()]
            buttons = [c for c in visible if c.element_info.control_type == 'Button'
                       and c.window_text().strip().casefold() in sends]
            if len(edits) != 1 or len(buttons) != 1:
                raise TargetError('INPUT_UNAVAILABLE')
            edit = edits[0]
            composer_names = {'message codex', 'ask codex anything', 'message', 'prompt',
                              'ask a follow-up', 'follow up', 'сообщение codex', 'сообщение'}
            if edit.window_text().strip().casefold() not in composer_names:
                raise TargetError('INPUT_UNAVAILABLE')
            if edit.element_info.element.CurrentIsPassword or edit.iface_value.CurrentIsReadOnly:
                raise TargetError('INPUT_UNAVAILABLE')
            if edit.iface_value.CurrentValue.strip():
                raise TargetError('INPUT_UNAVAILABLE')
            # A ValuePattern is mandatory: contenteditable implementations lacking it fail closed.
            return edit, buttons[0]
        except TargetError:
            raise
        except Exception:
            raise TargetError('INPUT_UNAVAILABLE') from None

    def valid_controls(self, window, controls, populated=False):
        try:
            edit, send = controls
            for control in controls:
                if not control.is_visible() or control.process_id() != window.pid:
                    return False
                if control.top_level_parent().handle != window.hwnd:
                    return False
            return (edit.is_enabled() and edit.has_keyboard_focus()
                    and not edit.iface_value.CurrentIsReadOnly
                    and (not populated or send.is_enabled()))
        except Exception:
            return False

    def set_text(self, controls, text):
        controls[0].iface_value.SetValue(text)
        return controls[0].iface_value.CurrentValue == text

    def submit(self, controls):
        controls[1].iface_invoke.Invoke()


class DesktopTarget:
    def __init__(self, backend, choose, log, journal, cancelled=lambda: False):
        self.backend, self.choose, self.log = backend, choose, log
        self.journal = Path(journal)
        self.selected = None
        self.lock = threading.Lock()
        self.cancelled = cancelled

    def validate(self):
        try:
            current = self.backend.describe(self.selected.hwnd)
            old = self.selected
            if not is_codex(current) or (current.pid, current.exe, current.created) != (old.pid, old.exe, old.created):
                raise TargetError('WINDOW_NOT_FOUND')
            return current
        except Exception:
            raise TargetError('WINDOW_NOT_FOUND') from None

    def resolve(self):
        if self.selected is not None:
            # A stale saved handle must never silently select a different conversation.
            return self.validate()
        windows = [w for w in self.backend.windows() if is_codex(w)]
        if not windows:
            raise TargetError('WINDOW_NOT_FOUND')
        chosen = windows[0] if len(windows) == 1 else self.choose(windows)
        if chosen not in windows:
            raise TargetError('AMBIGUOUS_WINDOW')
        self.selected = chosen
        return self.validate()

    def guard(self, window, controls, populated=False, check_cancel=True):
        if check_cancel and self.cancelled():
            raise TargetError('STOPPED')
        self.validate()
        if self.backend.foreground() != window.hwnd:
            raise TargetError('ACTIVATION_FAILED')
        if not self.backend.valid_controls(window, controls, populated):
            raise TargetError('INPUT_UNAVAILABLE')

    def prepare(self, check_cancel=True):
        window = self.resolve()
        self.log(f'Target process={PureWindowsPath(window.exe).name} pid={window.pid} '
                 f'HWND={window.hwnd} title={window.title!r} executable={window.exe}')
        active = self.backend.activate(window)
        self.log(f'Activation HWND={window.hwnd} result={active} foreground={self.backend.foreground()}')
        if not active or self.backend.foreground() != window.hwnd:
            raise TargetError('ACTIVATION_FAILED')
        self.validate()
        controls = self.backend.controls(window)
        self.guard(window, controls, check_cancel=check_cancel)
        return window, controls

    def run(self, prompt=None, operation='manual'):
        if not self.lock.acquire(blocking=False):
            return False, 'DELIVERY_PENDING'
        reserved = False
        initialized = False
        try:
            if hasattr(self.backend, 'begin'):
                self.backend.begin()
                initialized = True
            # A durable exclusive reservation prevents replay, even across controller restarts.
            if prompt is not None and self.journal.exists():
                raise TargetError('DELIVERY_PENDING')
            window, controls = self.prepare(check_cancel=prompt is not None)
            if prompt is None:
                return True, f'FOUND|{window.pid}|{window.hwnd}|{window.title}'
            record = {'pid': window.pid, 'hwnd': window.hwnd, 'state': 'delivery_pending',
                      'operation': operation, 'digest': hashlib.sha256(prompt.encode()).hexdigest()}
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open('x', encoding='utf-8') as stream:
                json.dump(record, stream)
                stream.flush()
                os.fsync(stream.fileno())
            reserved = True
            self.guard(window, controls)
            if not self.backend.set_text(controls, prompt):
                raise TargetError('DELIVERY_UNCONFIRMED')
            self.guard(window, controls, populated=True)
            self.backend.submit(controls)
            return True, f'SENT|{window.pid}|{window.hwnd}|{window.title}'
        except TargetError as exc:
            self.log(f'Desktop error={exc.code}; next attempt=' + ('blocked pending delivery review' if reserved or self.journal.exists() else 'after target/input correction'))
            return False, exc.code
        except Exception:
            code = 'DELIVERY_UNCONFIRMED' if reserved else 'INPUT_UNAVAILABLE'
            self.log(f'Desktop error={code}; next attempt=manual review required')
            return False, code
        finally:
            try:
                if initialized:
                    self.backend.end()
            finally:
                self.lock.release()

    def completed(self, operation):
        with self.lock:
            try:
                record = json.loads(self.journal.read_text(encoding='utf-8'))
                if record.get('operation') == operation:
                    self.journal.unlink()
            except (OSError, ValueError):
                pass


class DesktopTargetMixin:
    def reset_desktop_delivery(self):
        from tkinter import messagebox
        if getattr(self, 'worker', None) and self.worker.is_alive():
            messagebox.showinfo('ACC', 'Stop ACC before changing the target or resolving delivery.')
            return
        if messagebox.askyesno('Resolve delivery', 'Have you checked the selected Codex conversation and confirmed that the previous message is completed or was never sent? This permits a new submission.'):
            target = self.desktop_target()
            with target.lock:
                target.journal.unlink(missing_ok=True)
                target.selected = None
            self.enqueue_log('Delivery explicitly resolved by user; next attempt requires target validation.')

    def desktop_target(self):
        if not hasattr(self, '_desktop_target'):
            self._desktop_target = DesktopTarget(WinBackend(), self.choose_codex_window,
                                                self.enqueue_log,
                                                Path.home() / '.acc_codex_delivery.json',
                                                lambda: self.stop_event.is_set())
        return self._desktop_target

    def choose_codex_window(self, windows):
        from tkinter import simpledialog
        completed = threading.Event()
        result = []
        def select():
            try:
                labels = '\n'.join(f'{i + 1}. {w.title} | PID {w.pid} | HWND {w.hwnd} | {w.exe}' for i, w in enumerate(windows))
                number = simpledialog.askinteger('Choose Codex window', labels, minvalue=1, maxvalue=len(windows), parent=self.root)
                result.append(windows[number - 1] if number else None)
            finally:
                completed.set()
        self.ui_queue.put(select)
        if not completed.wait(60):
            return None
        return result[0] if result else None

    def find_desktop_codex_window(self):
        try:
            return self.desktop_target().run()
        except Exception:
            return False, 'INPUT_UNAVAILABLE'

    def test_desktop_codex(self):
        ok, info = self.find_desktop_codex_window()
        self.enqueue_log(f'Test Codex: {info}; no message sent.')

    def activate_and_send_to_desktop(self, prompt):
        if self.stop_event.is_set():
            return False, 'STOPPED'
        try:
            task = getattr(self, 'current_task', None)
            target = self.desktop_target()
            if task and (target.selected is None or target.selected.hwnd != task.window_handle
                         or target.selected.pid != task.window_pid):
                return False, 'WINDOW_NOT_FOUND'
            return target.run(prompt, str(task.task_start_ns) if task else 'manual')
        except Exception:
            return False, 'INPUT_UNAVAILABLE'
