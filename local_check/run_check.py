"""Локальная проверка агента — упрощённое подобие демо-проверки AgentScore.

Запуск из корня репозитория агента:
    python local_check/run_check.py

Что делает:
  1) копирует local_check/sample/repo во временную папку и делает из неё git-репозиторий;
  2) запускает агента так же, как платформа: --task-file ... --workspace ...;
  3) печатает git diff — именно его оценивает платформа;
  4) подкладывает скрытые тесты и запускает их.
Модель и ключ берутся из окружения или файла .env: OPENAI_API_KEY, OPENAI_MODEL
(и OPENAI_BASE_URL, если нужен не api.openai.com).
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "local_check" / "sample"
RUN = ROOT / "local_check" / ".run"


def load_env_file():
    """Минимальное чтение .env, чтобы не зависеть от python-dotenv."""
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def main() -> int:
    load_env_file()
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.environ.get(k)]
    if missing:
        print("Не заданы переменные:", ", ".join(missing))
        return 1

    shutil.rmtree(RUN, ignore_errors=True)
    testbed, task_dir = RUN / "testbed", RUN / "task"
    shutil.copytree(SAMPLE / "repo", testbed)
    task_dir.mkdir(parents=True)
    shutil.copy(SAMPLE / "TASK.md", task_dir / "TASK.md")
    git("init", "-q", cwd=testbed)
    git("add", "-A", cwd=testbed)
    git("-c", "user.name=check", "-c", "user.email=check@local", "commit", "-qm", "init", cwd=testbed)

    cmd = [sys.executable, str(ROOT / "agent.py"),
           "--task-file", str(task_dir / "TASK.md"), "--workspace", str(testbed)]
    print("$", " ".join(cmd))
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=testbed, timeout=600)  # на платформе тоже 600 секунд
    except subprocess.TimeoutExpired:
        print("Агент не уложился в 600 секунд")
        return 1
    print(f"\nкод завершения {proc.returncode}, время {time.perf_counter() - start:.0f} с")

    diff = git("diff", cwd=testbed)
    print("\n===== git diff (это оценивает платформа) =====")
    print(diff or "(пусто — пустой патч получает 0 баллов)")

    for test in (SAMPLE / "hidden_tests").iterdir():
        shutil.copy(test, testbed / test.name)
    print("===== скрытые тесты =====")
    tests = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=testbed,
                           capture_output=True, text=True)
    print(tests.stderr[-2000:])
    ok = proc.returncode == 0 and diff and tests.returncode == 0
    print("ИТОГ:", "задача решена" if ok else "задача не решена")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
