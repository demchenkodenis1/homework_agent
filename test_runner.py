"""Запуск выбранных pytest/unittest без shell, с лимитом времени и памяти вывода."""
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

MAX_TEST_SECONDS = 60
OUTPUT_BYTES = 65536


def python_command():
    explicit = os.environ.get("AGENT_TEST_PYTHON")
    if explicit:
        return [explicit]
    conda = shutil.which("conda") if os.name != "nt" else None
    if conda:
        return [conda, "run", "--no-capture-output", "-n", "testbed", "python"]
    return [sys.executable]


def stop_process(proc):
    if os.name == "nt":
        if proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if proc.poll() is None:
                proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_selected(root, target, runner, timeout, max_chars=4000):
    """target уже проверен Workspace; сохраняем конец лога с итогом тестов."""
    command = python_command()
    if runner == "pytest":
        command += ["-m", "pytest", "-q", "--tb=short", "--color=no", "-p", "no:cacheprovider", "--", target]
    else:
        from pathlib import Path
        path = Path(target)
        folder, pattern = (path.parent, path.name) if path.is_file() else (path, "test*.py")
        command += ["-m", "unittest", "discover", "-s", str(folder), "-p", pattern, "-v"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=os.name != "nt",
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as exc:
        return dict(status="ERROR", exit_code=None, seconds=0), f"ERROR: запуск тестов не состоялся ({type(exc).__name__})"
    tail = bytearray()
    total = 0

    def drain():
        nonlocal total
        try:
            while data := proc.stdout.read1(4096):
                total += len(data)
                tail.extend(data)
                if len(tail) > OUTPUT_BYTES:
                    del tail[:-OUTPUT_BYTES]
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        stop_process(proc)
        proc.wait(timeout=5)
        reader.join(timeout=1)
        if not reader.is_alive():
            proc.stdout.close()
    text = bytes(tail).decode("utf-8", errors="replace")
    if timed_out:
        status = "TIMEOUT"
    elif re.search(r"Ran 0 tests\b", text) or (runner == "pytest" and proc.returncode == 5):
        status = "NO_TESTS"
    elif runner == "unittest" and not re.search(r"Ran [1-9][0-9]* tests?\b", text):
        status = "ERROR"
    elif proc.returncode == 0:
        status = "PASSED"
    elif runner == "pytest" and proc.returncode != 1:
        status = "ERROR"
    else:
        status = "FAILED"
    record = dict(status=status, exit_code=proc.returncode, seconds=round(time.monotonic() - started, 3))
    header = ("" if status == "PASSED" else "ERROR ") + (
        f"run_tests: {status}; exit_code={proc.returncode}; seconds={record['seconds']}; limit={timeout:.1f}s")
    lines = text.splitlines()
    body = "\n".join(lines[-48:])
    available = max(0, max_chars - len(header) - 80)
    shortened = total > len(tail) or len(lines) > 48 or len(body) > available
    body = body[-available:] if available else ""
    marker = "\n[показан конец вывода тестов; начало сокращено]" if shortened else ""
    return record, header + marker + ("\n" + body if body else "")
