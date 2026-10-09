import os
import tkinter as tk
from tkinter import messagebox

import pyttsx3

from auth import TeacherStore
from monitor import ATTENDANCE_REPORT_FILE, DATA_DIR, run_monitoring


TEACHER_DB = os.path.join(DATA_DIR, "teacher.db")


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
# GUI
# -------------------------
store = TeacherStore(TEACHER_DB)


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
    if store.has_teacher():
        login_window()
    else:
        create_account_window()
