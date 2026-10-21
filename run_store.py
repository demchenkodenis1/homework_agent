"""Необязательный локальный журнал. Без пути SQLite не открывается."""
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path, task, model, workspace, context_mode, search_mode):
        self.connection = None
        self.run_id = str(uuid.uuid4()) if path else None
        if not path:
            return
        target = Path(path).expanduser().resolve()
        source = Path(__file__).resolve().parent
        if target.is_relative_to(Path(workspace).resolve()) or target.is_relative_to(source):
            raise ValueError("База должна лежать вне workspace и каталога агента")
        try:
            import sqlite3
            target.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(target, timeout=0.2)
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                    status TEXT NOT NULL, model TEXT, context_mode TEXT, search_mode TEXT,
                    workspace TEXT, task_hash TEXT, task_preview TEXT,
                    agent_commit TEXT, source_hash TEXT, summary_json TEXT);
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    model_step INTEGER NOT NULL, kind TEXT NOT NULL, started_at TEXT,
                    status TEXT, tool TEXT, call_id TEXT, arguments TEXT, output TEXT,
                    seconds REAL, input_tokens INTEGER, output_tokens INTEGER,
                    history_chars INTEGER, reused INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    step_id INTEGER NOT NULL REFERENCES steps(id), runner TEXT, target TEXT,
                    status TEXT, exit_code INTEGER, seconds REAL, outdated INTEGER DEFAULT 0,
                    output TEXT);
                CREATE INDEX IF NOT EXISTS steps_by_run ON steps(run_id, model_step);
                CREATE INDEX IF NOT EXISTS tests_by_run ON test_runs(run_id);
            ''')
            try:
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                                        capture_output=True, text=True, timeout=2, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                commit = None
            digest = hashlib.sha256()
            for name in ("agent.py", "context_support.py", "search_support.py", "test_runner.py", "run_store.py", "agent.json"):
                file = source / name
                if file.is_file():
                    digest.update(name.encode() + b"\0" + file.read_bytes())
            self.connection.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                self.run_id, now(), None, "running", model, context_mode, search_mode,
                str(workspace), hashlib.sha256(task.encode()).hexdigest(), task[:2000],
                commit, digest.hexdigest(), None))
            self.connection.commit()
        except Exception as exc:
            self.disable(exc)

    def disable(self, exc):
        print(f"SQLITE_LOG_DISABLED: {type(exc).__name__}; агент продолжает без базы", file=sys.stderr)
        if self.connection:
            self.connection.close()
        self.connection = None

    def write(self, operation):
        if self.connection is None:
            return None
        try:
            with self.connection:
                return operation(self.connection)
        except Exception as exc:
            self.disable(exc)
            return None

    def request(self, step, history_chars):
        return self.write(lambda db: db.execute(
            "INSERT INTO steps(run_id,model_step,kind,started_at,status,history_chars) VALUES (?,?,?,?,?,?)",
            (self.run_id, step, "model", now(), "running", history_chars)).lastrowid)

    def response(self, row, status, seconds, usage=None):
        self.write(lambda db: db.execute(
            "UPDATE steps SET status=?,seconds=?,input_tokens=?,output_tokens=? WHERE id=?",
            (status, seconds, getattr(usage, "input_tokens", None),
             getattr(usage, "output_tokens", None), row)))

    def tool(self, step, name, call_id, arguments, output, seconds, reused, test=None):
        def save(db):
            row = db.execute(
                "INSERT INTO steps(run_id,model_step,kind,started_at,status,tool,call_id,arguments,output,seconds,reused) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, step, "tool", now(), "error" if output.startswith("ERROR") else "returned",
                 name, call_id, arguments[:4000], output[:4000], seconds, int(reused))).lastrowid
            if name == "replace_in_file" and not reused and not output.startswith("ERROR"):
                db.execute("UPDATE test_runs SET outdated=1 WHERE run_id=?", (self.run_id,))
            if test is not None:
                db.execute(
                    "INSERT INTO test_runs(run_id,step_id,runner,target,status,exit_code,seconds,outdated,output) VALUES (?,?,?,?,?,?,?,?,?)",
                    (self.run_id, row, test["runner"], test["target"], test["status"], test["exit_code"],
                     test["seconds"], int(test["outdated"]), output[:4000]))
        self.write(save)

    def finish(self, stats):
        self.write(lambda db: db.execute(
            "UPDATE runs SET finished_at=?,status=?,summary_json=? WHERE id=?",
            (now(), stats["status"], json.dumps(stats, ensure_ascii=False), self.run_id)))
        if self.connection:
            self.connection.close()
            self.connection = None
