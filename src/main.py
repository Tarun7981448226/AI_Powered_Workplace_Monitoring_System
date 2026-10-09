"""Step 4 - the attendance system: teacher login, then a timed monitoring session.

Run this file. The first launch asks you to create the teacher account; after
that, log in, choose the session length and monitoring starts (ESC stops early).
The report is saved to data/attendance_report.json.
"""
import hashlib
import hmac
import os
import sqlite3
import tkinter as tk
from tkinter import messagebox

import pyttsx3

from face_recognizer import DATA_DIR, run_monitoring


TEACHER_DB = os.path.join(DATA_DIR, "teacher.db")
HASH_PREFIX = "pbkdf2"


# -------------------------
# Voice system
# -------------------------
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
except Exception:  # no speech engine available
    engine = None


def speak(text):
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# -------------------------
# Teacher accounts (salted PBKDF2 password hashes)
# -------------------------
def hash_password(password, salt=None, iterations=200_000):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{HASH_PREFIX}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        prefix, iterations, salt_hex, digest_hex = stored.split("$")
        if prefix != HASH_PREFIX:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except ValueError:
        return False


class TeacherStore:
    """SQLite store for the teacher account. Old plain-text accounts still work
    and are upgraded to a hash on their first login."""

    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS teacher(email TEXT, password TEXT)")
        self.conn.commit()

    def has_teacher(self):
        return self.conn.execute("SELECT 1 FROM teacher LIMIT 1").fetchone() is not None

    def create(self, email, password):
        email = email.strip().lower()
        if "@" not in email:
            raise ValueError("Enter a valid email address")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        if self.conn.execute("SELECT 1 FROM teacher WHERE email=?", (email,)).fetchone():
            raise ValueError("An account with this email already exists")
        self.conn.execute("INSERT INTO teacher VALUES(?,?)", (email, hash_password(password)))
        self.conn.commit()

    def verify(self, email, password):
        email = email.strip().lower()
        row = self.conn.execute("SELECT password FROM teacher WHERE email=?", (email,)).fetchone()
        if row is None:
            return False

        stored = row[0]
        if stored.startswith(HASH_PREFIX + "$"):
            return verify_password(password, stored)

        if hmac.compare_digest(stored, password):  # legacy plain-text account
            self.conn.execute("UPDATE teacher SET password=? WHERE email=?",
                              (hash_password(password), email))
            self.conn.commit()
            return True
        return False


store = None  # created in __main__ (tests replace it)


# -------------------------
# Windows
# -------------------------
def create_account_window():
    root = tk.Tk()
    root.title("Create Teacher Account")
    root.geometry("350x240")

    tk.Label(root, text="Create Teacher Account", font=("Arial", 14)).pack(pady=10)

    tk.Label(root, text="Email").pack()
    email_entry = tk.Entry(root, width=30)
    email_entry.pack()

    tk.Label(root, text="Password (min 6 characters)").pack()
    pass_entry = tk.Entry(root, show="*", width=30)
    pass_entry.pack()

    def create_account():
        try:
            store.create(email_entry.get(), pass_entry.get())
        except ValueError as error:
            messagebox.showerror("Error", str(error))
            return
        messagebox.showinfo("Success", "Account created")
        root.destroy()
        login_window()

    tk.Button(root, text="Create Account", command=create_account).pack(pady=10)
    root.mainloop()


def login_window():
    root = tk.Tk()
    root.title("AI Powered Workplace Monitoring System")
    root.geometry("350x320")

    tk.Label(root, text="Teacher Login", font=("Arial", 16)).pack(pady=10)

    tk.Label(root, text="Email").pack()
    email_entry = tk.Entry(root, width=30)
    email_entry.pack()

    tk.Label(root, text="Password").pack()
    pass_entry = tk.Entry(root, show="*", width=30)
    pass_entry.pack()

    tk.Label(root, text="Session Time (HH : MM : SS)").pack(pady=10)

    time_frame = tk.Frame(root)
    time_frame.pack()

    hour_spin = tk.Spinbox(time_frame, from_=0, to=23, width=5)
    hour_spin.pack(side="left")
    tk.Label(time_frame, text=":").pack(side="left")
    min_spin = tk.Spinbox(time_frame, from_=0, to=59, width=5)
    min_spin.pack(side="left")
    tk.Label(time_frame, text=":").pack(side="left")
    sec_spin = tk.Spinbox(time_frame, from_=0, to=59, width=5)
    sec_spin.pack(side="left")

    def login():
        if not store.verify(email_entry.get(), pass_entry.get()):
            speak("Wrong password try again")
            messagebox.showerror("Error", "Wrong Credentials")
            return

        try:
            total_seconds = (int(hour_spin.get()) * 3600 + int(min_spin.get()) * 60
                             + int(sec_spin.get()))
        except ValueError:
            messagebox.showerror("Error", "Session time must be numbers")
            return

        if total_seconds == 0:
            messagebox.showerror("Error", "Set monitoring time")
            return

        speak("Permission granted")
        root.destroy()

        try:
            run_monitoring(total_seconds, speak=speak)
        except FileNotFoundError as error:
            print("[ERROR]", error)
            speak("No trained model found")

    tk.Button(root, text="Login", command=login).pack(pady=20)
    root.mainloop()


if __name__ == "__main__":
    store = TeacherStore(TEACHER_DB)

    if store.has_teacher():
        login_window()
    else:
        create_account_window()
