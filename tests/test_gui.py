"""Drives the real Tkinter screens in main.py (clicks are simulated, camera is stubbed).

    python tests/test_gui.py
"""
import os
import sys
import tempfile
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import main
from auth import TeacherStore


def widgets(root, kind):
    out, stack = [], [root]
    while stack:
        w = stack.pop(0)
        stack += w.winfo_children()
        if isinstance(w, kind):
            out.append(w)
    return out


def fill(entry, text):
    entry.delete(0, tk.END)
    entry.insert(0, text)


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        main.store = TeacherStore(os.path.join(self.tmp.name, "t.db"))
        self.errors, self.infos, self.sessions = [], [], []
        self.patches = [
            mock.patch.object(main.messagebox, "showerror", lambda t, m: self.errors.append(m)),
            mock.patch.object(main.messagebox, "showinfo", lambda t, m: self.infos.append(m)),
            mock.patch.object(main, "speak", lambda text: None),
            mock.patch.object(main, "run_monitoring",
                              lambda secs, **kw: self.sessions.append(secs)),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_window(self, builder, interact):
        """Runs builder() with mainloop replaced by interact(root)."""
        def fake_mainloop(root):
            interact(root)
            try:
                root.destroy()
            except tk.TclError:
                pass
        with mock.patch.object(tk.Tk, "mainloop", fake_mainloop):
            builder()

    def button(self, root, label):
        return next(b for b in widgets(root, tk.Button) if b["text"] == label)

    def test_password_fields_are_masked(self):
        def interact(root):
            self.assertTrue(all(e.cget("show") == "*" for e in widgets(root, tk.Entry)[1:]))
        self.run_window(main.create_account_window, interact)

    def test_create_account_rejects_bad_input_then_accepts_good(self):
        def interact(root):
            email, pw = widgets(root, tk.Entry)
            fill(email, "bad"); fill(pw, "secret1")
            self.button(root, "Create Account").invoke()
            self.assertFalse(main.store.has_teacher())
            self.assertTrue(self.errors)
        self.run_window(main.create_account_window, interact)

    def test_full_flow_create_wrong_password_then_login_starts_session(self):
        def create(root):
            email, pw = widgets(root, tk.Entry)
            fill(email, "teacher@school.com"); fill(pw, "secret1")
            with mock.patch.object(main, "login_window", lambda: None):
                self.button(root, "Create Account").invoke()
        self.run_window(main.create_account_window, create)
        self.assertTrue(main.store.has_teacher())
        self.assertIn("Account created", self.infos)

        def wrong(root):
            email, pw = widgets(root, tk.Entry)
            fill(email, "teacher@school.com"); fill(pw, "WRONG")
            self.button(root, "Login").invoke()
        self.run_window(main.login_window, wrong)
        self.assertIn("Wrong Credentials", self.errors)
        self.assertEqual(self.sessions, [])

        def no_time(root):
            email, pw = widgets(root, tk.Entry)
            fill(email, "teacher@school.com"); fill(pw, "secret1")
            self.button(root, "Login").invoke()
        self.run_window(main.login_window, no_time)
        self.assertIn("Set monitoring time", self.errors)
        self.assertEqual(self.sessions, [])

        def good(root):
            email, pw = widgets(root, tk.Entry)
            fill(email, "teacher@school.com"); fill(pw, "secret1")
            h, m, s = widgets(root, tk.Spinbox)
            for spin, v in ((h, "0"), (m, "1"), (s, "30")):
                fill(spin, v)
            self.button(root, "Login").invoke()
        self.run_window(main.login_window, good)
        self.assertEqual(self.sessions, [90])


if __name__ == "__main__":
    unittest.main(verbosity=2)
