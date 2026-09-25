"""Readiness-aware wait aging: credit starts when the head job is truly ready.

Covers the fix for dependency-unlock aging settlement: a head-of-line job
queued for a long time behind an unfinished prerequisite must not earn AGING
credit for the blocked period when no batch ran meanwhile. The wait base is
floored at the head job's readiness instant:

* completion wiring: the LAST prerequisite's ``completed_at`` releases the
  dependent (multiple prerequisites -> the latest completion);
* revision wiring: a dependency revision that makes a blocked job ready
  releases it at the revision instant;
* the readiness instant is combined with the existing aging settlement
  point by taking the LATER of the two;
* ``preview_batch`` and ``run_batch`` share the computation, so identical
  state and reference instant yield identical ledger amounts, and the
  preview persists nothing.
"""
from __future__ import annotations

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus=1, minutes=5, weight=1.0, when=None, depends_on=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler.get_job(job["id"])


def aging_for(result, tenant):
    return [
        e for e in result["events"]
        if e["tenant_id"] == tenant and e["kind"] == "AGING"
    ]


def balance(tenant):
    return scheduler.get_tenant(tenant)["balance_gpu_min"]


# ---------------------------------------------------------------------------
# Completion wiring
# ---------------------------------------------------------------------------

def test_headline_case_credit_starts_at_prerequisite_completion(conn):
    # Head queued 60 min; its prerequisite completes 1 min before the batch;
    # no batch runs meanwhile. The old ledger granted the full 60 minutes.
    p = submit("a", when=iso(0))                          # J000001
    dep = submit("b", when=iso(0), depends_on=[p["id"]])  # J000002
    scheduler.run_batch(now=at(0))       # dispatches the prerequisite only
    scheduler.complete_job(p["id"], now=at(59))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert len(aging) == 1
    assert aging[0]["amount"] == pytest.approx(1.0)
    assert aging[0]["detail"] == "waited 1.000 min"
    # b: 1 aging + 60 quantum - 5 cost = 56, and the job dispatches.
    assert r["selection_order"] == ["J000002"]
    assert balance("b") == pytest.approx(56.0)


def test_completion_time_read_from_completed_at_column(conn):
    # The readiness instant derives from the stored completed_at (the same
    # value SQL-level pinning sets), not from any side clock.
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(p["id"])  # wall clock; pinned below
    db.get_conn().execute(
        "UPDATE jobs SET completed_at = ? WHERE id = ?", (iso(59), p["id"])
    )
    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


def test_multiple_prerequisites_release_at_last_completion(conn):
    p1 = submit("a", when=iso(0))
    p2 = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p1["id"], p2["id"]])
    scheduler.run_batch(now=at(0))                     # p1, p2 RUNNING
    scheduler.complete_job(p1["id"], now=at(10))     # still blocked by p2
    scheduler.complete_job(p2["id"], now=at(59))     # releases the dependent

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


def test_settlement_point_and_release_take_the_later(conn):
    # Batches at t=0 and t=30 advance b's checkpoint while blocked; the
    # prerequisite completes at t=59 -> credit from t=59 (the later of the
    # checkpoint t=30 and the release t=59), not from t=30 or t=0.
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.run_batch(now=at(30))
    assert scheduler.get_tenant("b")["aging_settled_at"] == iso(30)
    scheduler.complete_job(p["id"], now=at(59))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


def test_completion_before_arrival_keeps_arrival_as_base(conn):
    # Depending on an already-COMPLETED job: the job is ready from arrival.
    p = submit("a", when=iso(0))
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(p["id"], now=at(10))
    dep = submit("b", when=iso(50), depends_on=[p["id"]])
    assert dep["blocked_by"] == []

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(10.0)


def test_no_dependency_job_aging_unchanged(conn):
    # Regression: a plain job still earns aging for its whole wait.
    submit("a", gpus=8, minutes=10, weight=4.0, when=iso(0))  # saturates
    submit("b", gpus=1, minutes=5, when=iso(0))               # waits
    scheduler.run_batch(now=at(0))
    scheduler.complete_job("J000001", now=at(59))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Revision wiring
# ---------------------------------------------------------------------------

def test_revision_unblocking_counts_from_revision_instant(conn):
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    # Detaching at t=50 releases the job: credit starts at t=50, not t=0.
    scheduler.revise_dependencies(dep["id"], [], now=at(50))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(10.0)


def test_revision_to_completed_prerequisites_counts_from_revision(conn):
    p = submit("a", when=iso(0))
    done = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(done["id"], now=at(10))   # done long ago
    # Blocked on p; revision swaps to the already-completed job at t=50.
    scheduler.revise_dependencies(dep["id"], [done["id"]], now=at(50))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    # The revision instant (t=50) releases the job, not done's t=10.
    assert aging and aging[0]["amount"] == pytest.approx(10.0)


def test_ready_to_blocked_revision_starts_fresh_readiness_period(conn):
    p = submit("a", when=iso(0))
    q = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(p["id"], now=at(10))      # dep ready at t=10
    # Re-blocked by a revision at t=20; released by q's completion at t=59.
    scheduler.revise_dependencies(dep["id"], [q["id"]], now=at(20))
    scheduler.complete_job(q["id"], now=at(59))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


def test_ready_to_ready_revision_preserves_established_readiness(conn):
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(p["id"], now=at(10))      # dep ready since t=10
    # Detaching while ready must not resurrect the pre-t=10 blocked wait,
    # nor push readiness forward to the revision instant.
    scheduler.revise_dependencies(dep["id"], [], now=at(20))

    r = scheduler.run_batch(now=at(60))
    aging = aging_for(r, "b")
    assert aging and aging[0]["amount"] == pytest.approx(50.0)


def test_revision_does_not_touch_tenant_aging_checkpoint(conn):
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))                   # checkpoint -> t=0
    before = scheduler.get_tenant("b")["aging_settled_at"]

    scheduler.revise_dependencies(dep["id"], [], now=at(50))
    after = scheduler.get_tenant("b")
    assert after["aging_settled_at"] == before
    assert after["balance_gpu_min"] == 0.0


# ---------------------------------------------------------------------------
# Preview / batch parity
# ---------------------------------------------------------------------------

def test_preview_and_batch_share_the_readiness_computation(conn):
    p = submit("a", when=iso(0))
    dep = submit("b", when=iso(0), depends_on=[p["id"]])
    scheduler.run_batch(now=at(0))
    scheduler.complete_job(p["id"], now=at(59))

    c = db.get_conn()
    ledger_rows = c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"]
    batch_rows = c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"]

    preview = scheduler.preview_batch(now=at(60))
    # Zero writes: ledger, batches, job status, balances, aging clock.
    assert c.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()["n"] \
        == ledger_rows
    assert c.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"] \
        == batch_rows
    assert scheduler.get_job(dep["id"])["status"] == "QUEUED"
    assert balance("b") == 0.0

    committed = scheduler.run_batch(now=at(60))
    # Same state and instant -> identical ledger amounts and balances.
    assert [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in preview["events"]
    ] == [
        (e["kind"], e["tenant_id"], e["amount"], e["balance_after"])
        for e in committed["events"]
    ]
    assert preview["selection_order"] == committed["selection_order"]
    aging = aging_for(committed, "b")
    assert aging and aging[0]["amount"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

def test_legacy_jobs_table_migrates_ready_at(tmp_path, monkeypatch):
    """A DB created before readiness-aware aging gets jobs.ready_at added
    (NULL for existing jobs), and the aging settlement keeps working."""
    import sqlite3

    from app import db as db_mod
    from app.config import get_settings

    legacy = tmp_path / "legacy_ready.db"
    c = sqlite3.connect(str(legacy))
    c.executescript(
        """
        CREATE TABLE tenants (
            id TEXT PRIMARY KEY, weight REAL NOT NULL, deficit REAL NOT NULL
                DEFAULT 0, aging_settled_at TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            now_ref TEXT NOT NULL, capacity_gpus INTEGER NOT NULL,
            used_gpus_before INTEGER NOT NULL, available_gpus INTEGER NOT NULL,
            rounds INTEGER NOT NULL, jobs_selected INTEGER NOT NULL
        );
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL REFERENCES tenants(id),
            gpus INTEGER NOT NULL, duration_min INTEGER NOT NULL,
            cost_gpu_min REAL NOT NULL, status TEXT NOT NULL,
            queued_at TEXT NOT NULL, scheduled_at TEXT, completed_at TEXT,
            batch_id INTEGER REFERENCES batches(id), run_index INTEGER
        );
        CREATE TABLE job_dependencies (
            job_id TEXT NOT NULL REFERENCES jobs(id),
            depends_on TEXT NOT NULL REFERENCES jobs(id),
            PRIMARY KEY (job_id, depends_on)
        );
        CREATE TABLE ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL REFERENCES tenants(id),
            batch_id INTEGER REFERENCES batches(id),
            job_id TEXT REFERENCES jobs(id),
            kind TEXT NOT NULL, amount REAL NOT NULL, balance_after REAL NOT NULL,
            detail TEXT, created_at TEXT NOT NULL
        );
        INSERT INTO tenants (id, weight, deficit, created_at, updated_at)
        VALUES ('a', 1.0, 0, '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00');
        INSERT INTO tenants (id, weight, deficit, created_at, updated_at)
        VALUES ('b', 1.0, 0, '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00');
        -- Prerequisite completed at t=59; dependent queued since t=0.
        INSERT INTO jobs (id, tenant_id, gpus, duration_min, cost_gpu_min,
                          status, queued_at, scheduled_at, completed_at)
        VALUES ('J000001', 'a', 1, 5, 5, 'COMPLETED',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:59:00+00:00');
        INSERT INTO jobs (id, tenant_id, gpus, duration_min, cost_gpu_min,
                          status, queued_at)
        VALUES ('J000002', 'b', 1, 5, 5, 'QUEUED',
                '2026-01-01T00:00:00+00:00');
        INSERT INTO job_dependencies (job_id, depends_on)
        VALUES ('J000002', 'J000001');
        """
    )
    c.commit()
    c.close()

    monkeypatch.setenv("DB_PATH", str(legacy))
    get_settings.cache_clear()
    db_mod.reset_for_tests()
    try:
        db_mod.connect(str(legacy))
        cols = {
            r["name"]
            for r in db_mod.get_conn().execute("PRAGMA table_info(jobs)")
        }
        assert "ready_at" in cols
        # The readiness floor applies on the migrated schema too: the
        # dependent earns 1 minute of aging, not 60.
        r = scheduler.run_batch(now=at(60))
        aging = [
            e for e in r["events"]
            if e["tenant_id"] == "b" and e["kind"] == "AGING"
        ]
        assert aging and aging[0]["amount"] == pytest.approx(1.0)
    finally:
        db_mod.reset_for_tests()
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_batch_after_late_completion_grants_no_blocked_wait(client):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    queued_at = (now - timedelta(minutes=60)).isoformat()
    p = client.post("/jobs", json={
        "tenant_id": "a", "gpus": 1, "duration_min": 5}).json()["id"]
    dep = client.post("/jobs", json={
        "tenant_id": "b", "gpus": 1, "duration_min": 5,
        "depends_on": [p]}).json()["id"]
    # Both jobs arrived 60 minutes ago; no batch ran meanwhile.
    from app import db as db_mod
    for jid in (p, dep):
        db_mod.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (queued_at, jid))
    # The prerequisite dispatches at arrival time and completes only now,
    # moments before the scheduling batch.
    client.post("/scheduler/batch", json={"now": queued_at})
    client.post(f"/jobs/{p}/complete")

    body = client.post("/scheduler/batch").json()
    assert body["selection_order"] == [dep]
    aging = [e for e in body["events"]
             if e["tenant_id"] == "b" and e["kind"] == "AGING"]
    # Sub-second of true readiness -> effectively no aging, certainly not
    # the 60 minutes the job sat behind the unfinished prerequisite.
    assert aging == [] or aging[0]["amount"] < 1.0
