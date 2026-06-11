"""
database.py — SQLite Database for AI Safety Surveillance System

Tables:
  persons      -> registered workers (ArUco ID -> name/role)
  violations   -> every violation event (start/end, who, what, where)
  daily_summary-> pre-aggregated counts for fast dashboard charts
  attendance   -> first ArUco detection per person per day (work log)
"""

import sqlite3
import os
from datetime import datetime

DB_PATH        = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System\safety_system.db"
SCREENSHOT_DIR = r"C:\Mwafy\uni\GP\Graduation project\AI_Master_System\screenshots"


class SafetyDB:

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    aruco_id  INTEGER PRIMARY KEY,
                    name      TEXT    NOT NULL,
                    role      TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       INTEGER REFERENCES persons(aruco_id),
                    person_name     TEXT,
                    violation_type  TEXT    NOT NULL,
                    zone_name       TEXT,
                    camera_position TEXT    DEFAULT 'moving',
                    started_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                    ended_at        TEXT,
                    duration_sec    REAL,
                    screenshot_path TEXT,
                    confidence      TEXT    DEFAULT 'HIGH',
                    reviewed        INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    summary_date   TEXT    NOT NULL,
                    person_id      INTEGER,
                    person_name    TEXT,
                    violation_type TEXT,
                    count          INTEGER DEFAULT 0,
                    PRIMARY KEY (summary_date, person_id, violation_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   INTEGER REFERENCES persons(aruco_id),
                    person_name TEXT    NOT NULL,
                    work_date   TEXT    NOT NULL,
                    first_seen  TEXT    NOT NULL,
                    UNIQUE(person_id, work_date)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_person  ON violations(person_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_type    ON violations(violation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_started ON violations(started_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_viol_review  ON violations(reviewed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_att_date     ON attendance(work_date)")

        print(f"[DB] Database ready: {self.db_path}")
        self._seed_persons()
        self.close_stale_violations()   # clean up any orphans from previous runs


    # ── Seed default persons ───────────────────────────────────────────
    def _seed_persons(self):
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0:
                conn.executemany(
                    "INSERT OR IGNORE INTO persons (aruco_id, name, role) VALUES (?,?,?)",
                    [
                        (0, "Mohamed",  "Engineer"),
                        (1, "Ahmed",   "Visitor"),
                        (2, "Khaled",  "Worker"),
                        (3, "Youssef", "Worker"),
                    ]
                )
                print("[DB] Default persons seeded.")

    def reset_all_data(self):
        """Delete all violations, attendance and summary data. Keeps persons."""
        with self._connect() as conn:
            conn.execute("DELETE FROM violations")
            conn.execute("DELETE FROM attendance")
            conn.execute("DELETE FROM daily_summary")
        print("[DB] All data reset.")
    # ── Persons ───────────────────────────────────────────────────────
    def get_person(self, aruco_id: int):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM persons WHERE aruco_id=?", (aruco_id,)
            ).fetchone()

    def add_person(self, aruco_id: int, name: str, role: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO persons (aruco_id, name, role) VALUES (?,?,?)",
                (aruco_id, name, role)
            )
        print(f"[DB] Person added/updated: {name} (ArUco {aruco_id})")

    def get_all_persons(self):
        with self._connect() as conn:
            return conn.execute("SELECT * FROM persons ORDER BY aruco_id").fetchall()

    # ── Attendance ────────────────────────────────────────────────────
    def log_attendance(self, person_id: int, person_name: str):
        """
        Call the FIRST time ArUco is detected for a person each day.
        INSERT OR IGNORE means second call same day does nothing.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO attendance (person_id, person_name, work_date, first_seen) VALUES (?,?,?,?)",
                (person_id, person_name, today, now)
            )
        print(f"[DB] Attendance: {person_name} first seen today at {now}")

    def get_attendance_today(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM attendance WHERE work_date=? ORDER BY first_seen",
                (today,)
            ).fetchall()

    def get_attendance_by_person(self, person_id: int, limit: int = 30):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM attendance WHERE person_id=? ORDER BY work_date DESC LIMIT ?",
                (person_id, limit)
            ).fetchall()

    def get_attendance_by_date(self, date: str):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM attendance WHERE work_date=? ORDER BY first_seen",
                (date,)
            ).fetchall()

    # ── Violations ────────────────────────────────────────────────────
    def log_violation_start(self, violation_type: str, person_id: int = None,
                             person_name: str = None, zone_name: str = None,
                             camera_position: str = "moving",
                             screenshot_path: str = None,
                             confidence: str = "HIGH") -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO violations
                   (person_id, person_name, violation_type, zone_name,
                    camera_position, screenshot_path, confidence)
                   VALUES (?,?,?,?,?,?,?)""",
                (person_id, person_name, violation_type, zone_name,
                 camera_position, screenshot_path, confidence)
            )
            vid = cur.lastrowid
        print(f"[DB] Violation START: {violation_type} | {person_name or 'Unknown'} | id={vid}")
        return vid

    def log_violation_end(self, violation_id):
        with self._connect() as conn:
            row = conn.execute("SELECT started_at FROM violations WHERE id=?", (violation_id,)).fetchone()
            if row:
                dur = (datetime.now() - datetime.fromisoformat(row[0])).total_seconds()
                if dur < 3.0:
                    # Too short — discard entirely
                    conn.execute("DELETE FROM violations WHERE id=?", (violation_id,))
                    print(f"[DB] Violation discarded ({dur:.1f}s < 3s)")
                    return
                conn.execute("UPDATE violations SET ended_at=?, duration_sec=? WHERE id=?",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(dur,1), violation_id))
                v = conn.execute("SELECT person_id,person_name,violation_type,started_at FROM violations WHERE id=?", (violation_id,)).fetchone()
                if v:
                    conn.execute(
                        "INSERT INTO daily_summary (summary_date,person_id,person_name,violation_type,count) VALUES (?,?,?,?,1) ON CONFLICT(summary_date,person_id,violation_type) DO UPDATE SET count=count+1",
                        (v["started_at"][:10], v["person_id"], v["person_name"], v["violation_type"]))
                print(f"[DB] Violation END: id={violation_id} | {dur:.1f}s")

    def has_active_violation(self, person_id, violation_type):
        """Check if an unended violation of this type already exists."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id FROM violations
                WHERE person_id IS ? AND violation_type=?
                AND ended_at IS NULL
                ORDER BY started_at DESC LIMIT 1""",
                (person_id, violation_type)
            ).fetchone()
            return row['id'] if row else None

    def attach_screenshot(self, violation_id: int, path: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE violations SET screenshot_path=? WHERE id=?", (path, violation_id)
            )


    # ── Dashboard queries ─────────────────────────────────────────────
    def get_violations_today(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE started_at LIKE ? ORDER BY started_at DESC",
                (f"{today}%",)
            ).fetchall()

    def get_violations_by_person(self, person_id: int, limit: int = 50):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE person_id=? ORDER BY started_at DESC LIMIT ?",
                (person_id, limit)
            ).fetchall()

    def get_violations_by_type(self, violation_type: str, limit: int = 50):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE violation_type=? ORDER BY started_at DESC LIMIT ?",
                (violation_type, limit)
            ).fetchall()

    def get_unreviewed(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM violations WHERE (person_id IS NULL OR confidence='LOW') AND reviewed=0 ORDER BY started_at DESC"
            ).fetchall()

    def mark_reviewed(self, violation_id: int, person_id: int = None):
        with self._connect() as conn:
            conn.execute(
                "UPDATE violations SET reviewed=1, person_id=? WHERE id=?",
                (person_id, violation_id)
            )

    def get_weekly_summary(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT summary_date, person_name, violation_type, count FROM daily_summary WHERE summary_date >= date('now','-7 days') ORDER BY summary_date, person_name"
            ).fetchall()

    def get_monthly_summary(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT summary_date, person_id, person_name, violation_type, count FROM daily_summary WHERE summary_date >= date('now','-30 days') ORDER BY summary_date, person_name"
            ).fetchall()

    def get_person_stats(self, person_id: int):
        with self._connect() as conn:
            total        = conn.execute("SELECT COUNT(*) as cnt FROM violations WHERE person_id=?", (person_id,)).fetchone()["cnt"]
            by_type      = conn.execute("SELECT violation_type, COUNT(*) as cnt FROM violations WHERE person_id=? GROUP BY violation_type", (person_id,)).fetchall()
            avg_dur      = conn.execute("SELECT AVG(duration_sec) as avg FROM violations WHERE person_id=? AND duration_sec IS NOT NULL", (person_id,)).fetchone()["avg"]
            days_present = conn.execute("SELECT COUNT(*) as cnt FROM attendance WHERE person_id=?", (person_id,)).fetchone()["cnt"]
        return {
            "total":        total,
            "by_type":      {r["violation_type"]: r["cnt"] for r in by_type},
            "avg_duration": round(avg_dur or 0, 1),
            "days_present": days_present,
        }

    def get_all_violations(self, limit: int = 200):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM violations ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def close_stale_violations(self, max_age_minutes=60):
        """Close any violations stuck open for more than max_age_minutes."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE violations
                SET ended_at=datetime('now','localtime'), duration_sec=0, reviewed=1
                WHERE ended_at IS NULL
                AND started_at < datetime('now','localtime',?)""",
                (f'-{max_age_minutes} minutes',)
            )
        print(f"[DB] Stale violations closed (>{max_age_minutes}min).")

# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    db = SafetyDB()

    print("\n── Persons ──")
    for p in db.get_all_persons():
        print(f"  ArUco {p['aruco_id']}: {p['name']} ({p['role']})")

    print("\n── Attendance (first ArUco detection) ──")
    db.log_attendance(0, "Boukda")
    db.log_attendance(1, "Ahmed")
    db.log_attendance(0, "Boukda")   # second call same day → ignored

    print("\n── Today's attendance ──")
    for a in db.get_attendance_today():
        print(f"  {a['person_name']} arrived at {a['first_seen']}")

    print("\n── Simulating violation ──")
    vid = db.log_violation_start(
        violation_type  = "NO_HELMET",
        person_id       = 0,
        person_name     = "Boukda",
        zone_name       = "danger_zone",
        camera_position = "tilted_right",
        confidence      = "HIGH",
    )
    import time; time.sleep(2)
    db.log_violation_end(vid)

    print("\n── Today's violations ──")
    for v in db.get_violations_today():
        print(f"  [{v['started_at']}] {v['violation_type']} | {v['person_name'] or 'Unknown'} | {v['duration_sec']}s")

    print("\n── Boukda's stats ──")
    s = db.get_person_stats(0)
    print(f"  Violations   : {s['total']}")
    print(f"  By type      : {s['by_type']}")
    print(f"  Avg duration : {s['avg_duration']}s")
    print(f"  Days present : {s['days_present']}")

    print("\n[DB] Test complete.")