"""Fast unit tests (no camera, no training). Run from the project root:

    python tests/test_units.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auth import TeacherStore, hash_password, verify_password
from detector import FaceDetector
from monitor import PresenceTracker, compute_report


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")
        self.store = TeacherStore(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hash_is_salted_and_verifies(self):
        a, b = hash_password("secret1"), hash_password("secret1")
        self.assertNotEqual(a, b)
        self.assertTrue(verify_password("secret1", a))
        self.assertFalse(verify_password("wrong", a))
        self.assertNotIn("secret1", a)

    def test_create_and_login(self):
        self.assertFalse(self.store.has_teacher())
        self.store.create("Teacher@School.com", "secret1")
        self.assertTrue(self.store.has_teacher())
        self.assertTrue(self.store.verify("teacher@school.com", "secret1"))
        self.assertFalse(self.store.verify("teacher@school.com", "nope"))
        self.assertFalse(self.store.verify("other@school.com", "secret1"))

    def test_password_not_stored_in_plaintext(self):
        self.store.create("a@b.com", "secret1")
        stored = sqlite3.connect(self.db).execute("SELECT password FROM teacher").fetchone()[0]
        self.assertNotEqual(stored, "secret1")

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.store.create("not-an-email", "secret1")
        with self.assertRaises(ValueError):
            self.store.create("a@b.com", "123")
        self.store.create("a@b.com", "secret1")
        with self.assertRaises(ValueError):
            self.store.create("A@B.com", "secret2")

    def test_legacy_plaintext_account_is_upgraded(self):
        conn = sqlite3.connect(self.db)
        conn.execute("INSERT INTO teacher VALUES(?,?)", ("old@b.com", "plainpw"))
        conn.commit()
        self.assertFalse(self.store.verify("old@b.com", "bad"))
        self.assertTrue(self.store.verify("old@b.com", "plainpw"))
        stored = conn.execute("SELECT password FROM teacher").fetchone()[0]
        self.assertTrue(stored.startswith("pbkdf2$"))
        self.assertTrue(self.store.verify("old@b.com", "plainpw"))


class ReportTests(unittest.TestCase):
    def test_present_absent_threshold(self):
        report = compute_report({"A": 80.0, "B": 79.9}, 100, enrolled=["A", "B", "C"])
        self.assertEqual(report, {"A": "Present", "B": "Absent", "C": "Absent"})

    def test_unknown_never_reported(self):
        t = PresenceTracker()
        t.update("Unknown", 0.0)
        t.update("_background_1", 0.1)
        self.assertEqual(dict(t.presence_time), {})

    def test_presence_accumulates_and_ignores_gaps(self):
        t = PresenceTracker()
        for now in [0.0, 0.5, 1.0, 1.5]:
            t.update("A", now)
        self.assertAlmostEqual(t.presence_time["A"], 1.5)
        t.update("A", 10.0)  # left for > 2 s: the gap is not counted
        self.assertAlmostEqual(t.presence_time["A"], 1.5)

    def test_smoothing_needs_consistent_votes(self):
        t = PresenceTracker()
        box = (100, 100, 120, 120)
        out = [t.stable_name(box, n, now=i * 0.1) for i, n in enumerate(["A", "A", "A"])]
        self.assertEqual(out, ["Unknown", "Unknown", "A"])
        flicker = PresenceTracker()
        out = [flicker.stable_name(box, n, now=i * 0.1)
               for i, n in enumerate(["A", "B", "A", "B", "C"])]
        self.assertEqual(out[-1], "Unknown")

    def test_moving_face_keeps_its_votes(self):
        t = PresenceTracker()
        out = [t.stable_name((100 + 30 * i, 100, 200, 200), "A", now=i * 0.1) for i in range(5)]
        self.assertEqual(out[-1], "A")  # the old fixed grid reset on every step

    def test_two_faces_are_tracked_separately(self):
        t = PresenceTracker()
        left, right = (50, 100, 150, 150), (700, 100, 150, 150)
        for i in range(3):
            a = t.stable_name(left, "A", now=i * 0.1)
            b = t.stable_name(right, "B", now=i * 0.1)
        self.assertEqual((a, b), ("A", "B"))


class DetectorTests(unittest.TestCase):
    def test_uses_neural_detector_and_handles_blank_frame(self):
        det = FaceDetector()
        self.assertEqual(det.name, "YuNet")
        self.assertEqual(det.detect(np.zeros((480, 640, 3), np.uint8)), [])
        self.assertEqual(det.detect(np.zeros((480, 640), np.uint8)), [])

    def test_haar_fallback_works(self):
        det = FaceDetector()
        det.yunet = None
        self.assertEqual(det.detect(np.zeros((480, 640, 3), np.uint8)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
