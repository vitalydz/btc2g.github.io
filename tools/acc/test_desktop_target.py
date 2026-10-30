import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

from acc_desktop_target import DesktopTarget, TargetError, Window, WinBackend, is_codex

EXE = r'C:\Program Files\WindowsApps\OpenAI.Codex_26.924.1866.0_x64__2p2nqsd0c76g0\app\ChatGPT.exe'
CODEX = Window(100, 42, EXE, 'ChatGPT', 'created-1')
SETTINGS = Window(200, 6820, r'C:\Windows\System32\ApplicationFrameHost.exe', 'Settings', 'created-2')
ACC = Window(300, 43, r'C:\Python\python.exe', 'ACC v1.3 - Codex', 'created-3')


class FakeBackend:
    def __init__(self):
        self.items = [CODEX, SETTINGS, ACC]
        self.active = SETTINGS.hwnd
        self.activation_ok = True
        self.input_ok = True
        self.writes = []
        self.submits = []
        self.after_write = lambda: None

    def windows(self): return self.items
    def describe(self, hwnd):
        return next(w for w in self.items if w.hwnd == hwnd)
    def foreground(self): return self.active
    def activate(self, window):
        if self.activation_ok: self.active = window.hwnd
        return self.activation_ok
    def controls(self, window):
        if not self.input_ok: raise TargetError('INPUT_UNAVAILABLE')
        return window.hwnd
    def valid_controls(self, window, controls, populated=False): return self.input_ok
    def set_text(self, controls, text):
        self.writes.append((controls, text))
        self.after_write()
        return True
    def submit(self, controls): self.submits.append(controls)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Path(self.tmp.name) / 'delivery.json'
        self.backend = FakeBackend()
        self.logs = []
        self.choose = Mock(return_value=None)
        self.target = DesktopTarget(self.backend, self.choose, self.logs.append, self.journal)

    def assert_no_send(self, code):
        ok, actual = self.target.run('private-message')
        self.assertFalse(ok)
        self.assertEqual(actual, code)
        self.assertEqual(self.backend.writes, [])
        self.assertEqual(self.backend.submits, [])

    def test_identity_does_not_depend_on_title(self):
        self.assertTrue(is_codex(CODEX))
        self.assertFalse(is_codex(SETTINGS))
        self.assertFalse(is_codex(ACC))
        self.assertFalse(is_codex(Window(3, 5, r'C:\Other\Codex.exe', 'Codex', '1')))
        self.assertFalse(is_codex(Window(3, 5, r'C:\Users\me\AppData\Local\OpenAI\Codex\bin\hash\codex.exe', 'Codex', '1')))

    def test_settings_and_acc_foreground_activate_only_codex(self):
        for foreground in (SETTINGS.hwnd, ACC.hwnd):
            self.backend.active = foreground
            ok, _ = self.target.run()
            self.assertTrue(ok)
            self.assertEqual(self.backend.active, CODEX.hwnd)
        self.assertEqual(self.backend.writes, [])

    def test_activation_failure_never_writes_or_claims_limit(self):
        self.backend.activation_ok = False
        self.assert_no_send('ACTIVATION_FAILED')
        self.assertFalse(any('limit' in line for line in self.logs))

    def test_closed_codex(self):
        self.backend.items = [SETTINGS, ACC]
        self.assert_no_send('WINDOW_NOT_FOUND')

    def test_multiple_windows_require_choice(self):
        second = Window(101, 42, EXE, 'Second chat', 'created-1')
        self.backend.items.append(second)
        self.assert_no_send('AMBIGUOUS_WINDOW')
        self.choose.assert_called_once()
        self.choose.return_value = second
        ok, _ = self.target.run('private-message')
        self.assertTrue(ok)
        self.assertEqual(self.backend.submits, [101])

    def test_reused_hwnd_is_rejected(self):
        self.target.run()
        self.backend.items[0] = Window(100, 6820, SETTINGS.exe, 'Codex', 'created-2')
        self.assert_no_send('WINDOW_NOT_FOUND')

    def test_restarted_process_same_pid_is_rejected(self):
        self.target.run()
        self.backend.items[0] = Window(100, 42, EXE, 'ChatGPT', 'created-new')
        self.assert_no_send('WINDOW_NOT_FOUND')

    def test_input_unavailable(self):
        self.backend.input_ok = False
        self.assert_no_send('INPUT_UNAVAILABLE')

    def test_foreground_change_before_write(self):
        original = self.backend.controls
        def moved(window):
            value = original(window)
            self.backend.active = SETTINGS.hwnd
            return value
        self.backend.controls = moved
        self.assert_no_send('ACTIVATION_FAILED')

    def test_foreground_change_after_write_never_submits_or_replays(self):
        self.backend.after_write = lambda: setattr(self.backend, 'active', ACC.hwnd)
        self.assertEqual(self.target.run('private-message'), (False, 'ACTIVATION_FAILED'))
        self.assertEqual(self.backend.writes, [(100, 'private-message')])
        self.assertEqual(self.backend.submits, [])
        restarted = DesktopTarget(self.backend, self.choose, self.logs.append, self.journal)
        self.assertEqual(restarted.run('private-message'), (False, 'DELIVERY_PENDING'))
        self.assertEqual(len(self.backend.writes), 1)

    def test_success_does_not_duplicate_or_log_content(self):
        self.assertTrue(self.target.run('private-message')[0])
        self.assertEqual(self.target.run('private-message'), (False, 'DELIVERY_PENDING'))
        self.assertEqual(self.backend.submits, [100])
        self.assertNotIn('private-message', '\n'.join(self.logs))
        self.assertNotIn('private-message', self.journal.read_text())

    def test_only_matching_completion_unlocks_next_task(self):
        self.target.run('private-message', 'task-1')
        self.target.completed('task-2')
        self.assertTrue(self.journal.exists())
        self.target.completed('task-1')
        self.assertFalse(self.journal.exists())

    def test_stop_before_write(self):
        self.target.cancelled = lambda: True
        self.assert_no_send('STOPPED')

    def test_test_and_send_share_preparation(self):
        self.target.prepare = Mock(wraps=self.target.prepare)
        self.target.run()
        self.target.run('private-message')
        self.assertEqual(self.target.prepare.call_count, 2)

    def test_minimized_window_restore_precedes_activation(self):
        backend = object.__new__(WinBackend)
        backend.gui = Mock()
        backend.gui.IsIconic.return_value = True
        backend.gui.GetForegroundWindow.return_value = 100
        self.assertTrue(backend.activate(CODEX))
        backend.gui.ShowWindow.assert_called_once_with(100, 9)
        names = [c[0] for c in backend.gui.method_calls]
        self.assertLess(names.index('ShowWindow'), names.index('SetForegroundWindow'))


if __name__ == '__main__':
    unittest.main()
