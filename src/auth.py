import hashlib
import hmac
import os
import sqlite3


ITERATIONS = 200_000
PREFIX = "pbkdf2"


def hash_password(password, salt=None, iterations=ITERATIONS):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{PREFIX}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        prefix, iterations, salt_hex, digest_hex = stored.split("$")
        if prefix != PREFIX:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, AttributeError):
        return False


class TeacherStore:
    """SQLite store for teacher/admin accounts with salted password hashes.

    Accounts created by older versions (plain-text passwords) still work and
    are upgraded to a hash the first time they log in.
    """

    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS teacher(email TEXT, password TEXT)")
        self.conn.commit()

    def has_teacher(self):
        return self.conn.execute("SELECT 1 FROM teacher LIMIT 1").fetchone() is not None

    def create(self, email, password):
        email = email.strip().lower()
        if not email or "@" not in email:
            raise ValueError("Enter a valid email address")
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        exists = self.conn.execute(
            "SELECT 1 FROM teacher WHERE email=?", (email,)).fetchone()
        if exists:
            raise ValueError("An account with this email already exists")
        self.conn.execute("INSERT INTO teacher VALUES(?,?)", (email, hash_password(password)))
        self.conn.commit()

    def verify(self, email, password):
        email = email.strip().lower()
        row = self.conn.execute(
            "SELECT password FROM teacher WHERE email=?", (email,)).fetchone()
        if row is None:
            return False

        stored = row[0]
        if stored.startswith(PREFIX + "$"):
            return verify_password(password, stored)

        # legacy plain-text account: accept once, then upgrade to a hash
        if hmac.compare_digest(stored, password):
            self.conn.execute("UPDATE teacher SET password=? WHERE email=?",
                              (hash_password(password), email))
            self.conn.commit()
            return True
        return False
