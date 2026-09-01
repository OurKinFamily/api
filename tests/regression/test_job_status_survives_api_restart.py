"""Regression: a job that finished fine was reported as "unknown".

Only the parent knew a job's exit code — it came from proc.wait() in a thread
owned by the API process. Restart the API mid-job (a --reload in dev, a deploy
in prod) and that thread dies with it: the run stays "running", and the reaper
later notices the PID is gone and marks it "unknown". The Backfill Face Count
run on 2026-09-01 completed successfully and still showed "unknown".

Fix: the job records its own exit code to <run_id>.exit from inside the shell,
so the result outlives the process that started it, and the reaper recovers the
real status instead of guessing.
"""
import pytest

from app.routers import jobs


class TestStatusForExitCode:
    @pytest.mark.parametrize("code,expected", [
        (0, "completed"),
        (2, "completed_with_warnings"),   # finished, some files errored
        (-15, "cancelled"),               # SIGTERM
        (143, "cancelled"),               # same signal, shell-reported
        (1, "failed"),
        (None, "unknown"),                # genuinely no record
    ])
    def test_maps_exit_code_to_status(self, code, expected):
        assert jobs._status_for(code) == expected


class TestExitCodeRecovery:
    def test_reads_the_code_the_job_wrote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
        (tmp_path / "abc123.exit").write_text("0")
        assert jobs._recover_exit_code("abc123") == 0

    def test_recovers_a_failure_too(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
        (tmp_path / "abc123.exit").write_text("1")
        assert jobs._status_for(jobs._recover_exit_code("abc123")) == "failed"

    def test_returns_none_when_the_job_never_got_that_far(self, tmp_path, monkeypatch):
        # Killed hard enough that even the shell didn't write — "unknown" is
        # then the honest answer.
        monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
        assert jobs._recover_exit_code("nope") is None

    def test_survives_a_garbled_exit_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jobs, "RUNS_DIR", tmp_path)
        (tmp_path / "abc123.exit").write_text("")
        assert jobs._recover_exit_code("abc123") is None


def test_command_is_wrapped_to_record_its_own_exit_code():
    """Guard the wrapping itself: the helpers are useless if the shell never
    writes the file."""
    from pathlib import Path
    src = Path(jobs.__file__).read_text()
    assert "wrapped = " in src, "command must be wrapped before Popen"
    assert "rc=$?" in src and "exit $rc" in src, (
        "wrapper must capture and re-raise the real exit code, not swallow it"
    )
    assert "wrapped, shell=True" in src, "Popen must run the wrapped command"
