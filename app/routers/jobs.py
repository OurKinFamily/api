"""
Job runner — define jobs in jobs.json, execute via subprocess, stream logs.

Runs and logs are stored under JOBS_DATA_DIR (default /photos/__data/workers).
The docker exec commands in jobs.json require /var/run/docker.sock mounted in
the API container — configure that in the deploy stack when ready.
"""

import json
import logging
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])

JOBS_CONFIG   = Path(__file__).parent.parent.parent / "jobs.json"
JOBS_DATA_DIR = Path(os.environ.get("JOBS_DATA_DIR", "/photos/__data/workers"))
RUNS_DIR      = JOBS_DATA_DIR / "runs"
LOGS_DIR      = JOBS_DATA_DIR / "logs"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

active_processes: dict[str, subprocess.Popen] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jobs() -> list:
    if not JOBS_CONFIG.exists():
        return []
    return json.loads(JOBS_CONFIG.read_text())


def _jobs_by_id() -> dict:
    return {j["id"]: j for j in load_jobs()}


def load_run(run_id: str) -> dict | None:
    path = RUNS_DIR / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_run(run: dict):
    (RUNS_DIR / f"{run['id']}.json").write_text(json.dumps(run, indent=2))


def all_runs() -> list:
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            runs.append(json.loads(f.read_text()))
        except Exception:
            pass
    return runs[:50]


def build_command(job: dict, params: dict) -> str:
    subs = {}
    for p in job.get("params", []):
        name = p["name"]
        val  = params.get(name, p.get("default", ""))
        if p["type"] == "flag":
            subs[name] = p["flag"] if val else ""
        else:
            subs[name] = str(val) if val is not None else ""
    return job["shell_command"].format(**subs)


def get_last_log_line(log_path: str) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return ""
            chunk = min(size, 4096)
            f.seek(-chunk, 2)
            lines = f.read(chunk).decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line[:200]
    except Exception:
        pass
    return ""


def _run_job(run_id: str, cmd: str, log_path: Path):
    run = load_run(run_id)
    run["status"]     = "running"
    run["started_at"] = datetime.now().isoformat(timespec="seconds")
    save_run(run)

    with open(log_path, "w") as lf:
        lf.write(f"$ {cmd}\n\n")
        lf.flush()
        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                executable="/bin/bash", preexec_fn=os.setsid,
            )
            active_processes[run_id] = proc
            run = load_run(run_id)
            run["pid"] = proc.pid
            save_run(run)
            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            lf.write(f"\n\nERROR: {e}\n")
            exit_code = -1

    active_processes.pop(run_id, None)
    run = load_run(run_id)
    run["status"]      = "completed" if exit_code == 0 else ("cancelled" if exit_code == -15 else "failed")
    run["exit_code"]   = exit_code
    run["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_run(run)


def _annotate_runs(runs: list) -> list:
    for r in runs:
        if r["id"] in active_processes:
            r["status"] = "running"
        elif r.get("status") == "running":
            pid = r.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except (ProcessLookupError, PermissionError):
                    r["status"] = "unknown"
                    r["finished_at"] = r.get("finished_at") or datetime.now().isoformat(timespec="seconds")
                    save_run(r)
        if r.get("status") in ("running", "unknown"):
            r["last_line"] = get_last_log_line(r.get("log_path", ""))
    return runs


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
async def list_jobs():
    return load_jobs()


@router.get("/runs")
async def list_runs():
    return _annotate_runs(all_runs())


@router.post("/runs", status_code=202)
async def start_run(body: dict):
    job_id = body.get("job_id")
    params = body.get("params", {})

    jobs = _jobs_by_id()
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job      = jobs[job_id]
    run_id   = __import__("uuid").uuid4().hex[:8]
    cmd      = build_command(job, params)
    log_path = LOGS_DIR / f"{run_id}.log"

    run = {
        "id":          run_id,
        "job_id":      job_id,
        "job_name":    job["name"],
        "status":      "queued",
        "params":      params,
        "command":     cmd,
        "log_path":    str(log_path),
        "started_at":  None,
        "finished_at": None,
        "exit_code":   None,
    }
    save_run(run)
    threading.Thread(target=_run_job, args=(run_id, cmd, log_path), daemon=True).start()
    return run


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = load_run(run_id)
    if not run:
        raise HTTPException(404)
    if run_id in active_processes:
        run["status"] = "running"
    return run


@router.get("/runs/{run_id}/log")
async def get_run_log(run_id: str, offset: int = Query(0)):
    run = load_run(run_id)
    if not run:
        raise HTTPException(404)
    log_path = Path(run["log_path"])
    if not log_path.exists():
        return {"content": "", "offset": 0, "done": False}
    with open(log_path, "r", errors="replace") as f:
        f.seek(offset)
        content = f.read()
    new_offset = offset + len(content.encode())
    done = run_id not in active_processes and run.get("status") in ("completed", "failed", "cancelled")
    return {"content": content, "offset": new_offset, "done": done}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    proc = active_processes.get(run_id)
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
    run = load_run(run_id)
    if run:
        run["status"]      = "cancelled"
        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_run(run)
    return {"ok": True}


@router.post("/runs/{run_id}/dismiss")
async def dismiss_run(run_id: str):
    run = load_run(run_id)
    if not run:
        raise HTTPException(404)
    run["status"]      = "completed"
    run["finished_at"] = run.get("finished_at") or datetime.now().isoformat(timespec="seconds")
    save_run(run)
    return {"ok": True}
