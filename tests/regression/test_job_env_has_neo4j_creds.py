"""Regression: every Neo4j-touching job died with "NEO4J_PASSWORD not set".

Jobs run via subprocess.Popen and inherited the API process environment. But
pydantic-settings reads .env into the Settings object, not into os.environ, so
the child saw no NEO4J_* variables at all — Face Clustering and Backfill Face
Count both failed the moment they were clicked.

Fix: the runner passes them explicitly from settings. Deliberately injected at
run time rather than written into jobs.json, which is tracked — a credential in
a tracked file is what caused the 2026-06-02 incident.
"""
from pathlib import Path

from app.routers import jobs


def test_job_env_carries_neo4j_credentials():
    env = jobs._job_env()
    assert env["NEO4J_URI"] == jobs.settings.neo4j_uri
    assert env["NEO4J_USER"] == jobs.settings.neo4j_user
    assert env["NEO4J_PASSWORD"] == jobs.settings.neo4j_password


def test_job_env_still_inherits_the_ambient_environment(monkeypatch):
    """PATH and friends must survive — the shell commands rely on them."""
    monkeypatch.setenv("OURKIN_TEST_MARKER", "present")
    assert jobs._job_env()["OURKIN_TEST_MARKER"] == "present"


def test_popen_actually_passes_the_env():
    """Guard the call site, not just the helper — the helper existed and was
    simply never wired in."""
    lines = Path(jobs.__file__).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "subprocess.Popen(" in l)
    # The call spans a handful of lines; scan to its closing paren.
    call = []
    for l in lines[start:]:
        call.append(l)
        if l.strip() == ")":
            break
    assert any("env=_job_env()" in l for l in call), (
        "job subprocess must be given _job_env(): " + "\n".join(call)
    )


def test_no_credential_literal_in_jobs_json():
    """jobs.json is tracked. Credentials belong in the injected env only."""
    jobs_json = (Path(jobs.__file__).parents[2] / "jobs.json").read_text()
    assert jobs.settings.neo4j_password not in jobs_json
