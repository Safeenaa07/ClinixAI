"""SQLite storage. AI output (analyses) and the doctor's decisions are kept in
separate tables so every change can be audited."""
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

from config import DB_PATH


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()


def init():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, pw_hash TEXT, salt TEXT, level TEXT);
        CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY, scan_type TEXT, path TEXT, quality TEXT, user TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY, scan_id TEXT, result TEXT, report TEXT, level TEXT, user TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, user TEXT, action TEXT,
                                             final_text TEXT, reason TEXT, finding TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, user TEXT, event TEXT, detail TEXT);
        """)
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            # Demo accounts. Change these passwords before showing the app to anyone else.
            for name, level in (("dr_junior", "junior"), ("dr_specialist", "specialist")):
                salt = secrets.token_hex(16)
                c.execute("INSERT INTO users VALUES(?,?,?,?)", (name, _hash("clinix123", salt), salt, level))


def check_login(username: str, password: str):
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and secrets.compare_digest(_hash(password, row["salt"]), row["pw_hash"]):
        return {"username": row["username"], "level": row["level"]}
    return None


def audit(user: str, event: str, detail: dict | None = None):
    with _db() as c:
        c.execute("INSERT INTO audit(ts,user,event,detail) VALUES(?,?,?,?)",
                  (time.time(), user, event, json.dumps(detail or {})))


def add_scan(scan_id, scan_type, path, quality, user):
    with _db() as c:
        c.execute("INSERT INTO scans VALUES(?,?,?,?,?,?)",
                  (scan_id, scan_type, str(path), json.dumps(quality), user, time.time()))


def get_scan(scan_id):
    with _db() as c:
        row = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
    return dict(row) if row else None


def add_analysis(analysis_id, scan_id, result, report, level, user):
    with _db() as c:
        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?)",
                  (analysis_id, scan_id, json.dumps(result), json.dumps(report), level, user, time.time()))


def get_analysis(analysis_id):
    with _db() as c:
        row = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return dict(row) if row else None


def update_report(analysis_id, report, level):
    with _db() as c:
        c.execute("UPDATE analyses SET report=?, level=? WHERE id=?",
                  (json.dumps(report), level, analysis_id))


def add_decision(analysis_id, user, action, final_text=None, reason=None, finding=None):
    with _db() as c:
        c.execute("INSERT INTO decisions(analysis_id,user,action,final_text,reason,finding,created) VALUES(?,?,?,?,?,?,?)",
                  (analysis_id, user, action, final_text, reason, finding, time.time()))


def history(user: str, limit: int = 50):
    with _db() as c:
        rows = c.execute("""
            SELECT a.id, a.created, a.level, s.scan_type,
                   (SELECT action FROM decisions d WHERE d.analysis_id=a.id ORDER BY d.id DESC LIMIT 1) AS last_action
            FROM analyses a JOIN scans s ON s.id=a.scan_id
            WHERE a.user=? ORDER BY a.created DESC LIMIT ?""", (user, limit)).fetchall()
    return [dict(r) for r in rows]


def analytics():
    with _db() as c:
        actions = {r["action"]: r["n"] for r in
                   c.execute("SELECT action, COUNT(*) n FROM decisions GROUP BY action")}
        total = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    return {"analyses": total, "decisions": actions}
