from __future__ import annotations

import os
import hmac
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .access_control import AccessLedger


DATA_DIR = Path(os.getenv("ACCESS_DATA_DIR", "/data"))
ledger = AccessLedger(
    DATA_DIR / "access_ledger.sqlite",
    DATA_DIR / "server_ed25519.pem",
    task_deadline_seconds=float(os.getenv("TASK_DEADLINE_SECONDS", "30")),
    mechanism_variant=os.getenv("MECHANISM_VARIANT", "full"),
    max_probation_tasks=int(os.getenv("MAX_PROBATION_TASKS", "20")),
    expected_experiment_nodes=int(os.getenv("EXPECTED_EXPERIMENT_NODES", "0")),
    min_completed_tasks=int(os.getenv("MIN_PROBATION_COMPLETED_TASKS", "5")),
    min_evidence_mass=float(os.getenv("MIN_EVIDENCE_MASS", "1.50")),
    min_limited_clean_updates=int(os.getenv("MIN_LIMITED_CLEAN_UPDATES", "2")),
    min_limited_observation_updates=int(os.getenv(
        "MIN_LIMITED_OBSERVATION_UPDATES",
        os.getenv("MIN_LIMITED_CLEAN_UPDATES", "2"),
    )),
    limited_aggregation_weight=float(os.getenv("LIMITED_AGGREGATION_WEIGHT", "0.25")),
    access_lease_full_updates=int(os.getenv("ACCESS_LEASE_FULL_UPDATES", "5")),
    source_verifier_public_key=os.getenv("SOURCE_VERIFIER_PUBLIC_KEY") or None,
    trust_estimator=os.getenv("TRUST_ESTIMATOR", "dirichlet_lcb"),
    incumbent_node_ids={
        node_id.strip()
        for node_id in os.getenv("INCUMBENT_NODE_IDS", "").split(",")
        if node_id.strip()
    },
)
app = FastAPI(title="Selection-aware FL node access controller", version="0.1.0")


class Registration(BaseModel):
    node_id: str = Field(min_length=1, max_length=128)
    profile: str = Field(min_length=1, max_length=64)
    public_key: str
    registration_signature: str


class Receipt(BaseModel):
    node_id: str
    task_id: str
    assignment_hash: str
    signature: str


class Result(BaseModel):
    node_id: str
    task_id: str
    quality: float = Field(ge=0.0, le=1.0)
    result_hash: str
    work_product: str
    shadow_update: dict | None = None
    signature: str


class FlowerEvent(BaseModel):
    node_id: str
    event_id: str
    event_type: str
    metrics_hash: str
    signature: str


class OnboardingFinished(BaseModel):
    node_id: str
    final_access_state: str
    signature: str


class FitCommitment(BaseModel):
    node_id: str
    server_round: int = Field(ge=1)
    update_hash: str
    parent_hash: str
    signature: str


class RevokeTraining(BaseModel):
    node_id: str
    server_round: int = Field(ge=1)
    reason: str


class AcceptFit(BaseModel):
    node_id: str
    server_round: int = Field(ge=1)
    update_hash: str


class AggregatedFit(BaseModel):
    node_id: str
    server_round: int = Field(ge=1)
    update_hash: str


class VerifiedEvidenceSource(BaseModel):
    node_id: str
    task_id: str
    source_id: str = Field(min_length=1, max_length=128)
    verifier_signature: str


def require_internal_token(provided: str | None) -> None:
    configured = os.getenv("FL_CONTROL_TOKEN", "")
    if not configured or not provided or not hmac.compare_digest(provided, configured):
        raise HTTPException(status_code=403, detail="unauthorized training controller")


@app.post("/internal/evidence/source")
def verified_evidence_source(
    request: VerifiedEvidenceSource, x_internal_token: str | None = Header(default=None),
) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.record_verified_source, request.node_id, request.task_id,
        request.source_id, request.verifier_signature,
    )


@app.post("/internal/fit/verify")
def verify_fit(request: FitCommitment, x_internal_token: str | None = Header(default=None)) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.verify_fit_commitment, request.node_id, request.server_round,
        request.update_hash, request.parent_hash, request.signature,
    )


@app.post("/internal/fit/revoke")
def revoke_fit(request: RevokeTraining, x_internal_token: str | None = Header(default=None)) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.revoke_training_access, request.node_id,
        request.server_round, request.reason,
    )


@app.post("/internal/fit/remove")
def remove_baseline_participant(
    request: RevokeTraining, x_internal_token: str | None = Header(default=None),
) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.remove_baseline_participant, request.node_id,
        request.server_round, request.reason,
    )


@app.post("/internal/fit/accept")
def accept_fit(request: AcceptFit, x_internal_token: str | None = Header(default=None)) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.accept_fit_update, request.node_id,
        request.server_round, request.update_hash,
    )


@app.post("/internal/fit/aggregated")
def aggregated_fit(
    request: AggregatedFit, x_internal_token: str | None = Header(default=None),
) -> dict:
    require_internal_token(x_internal_token)
    return call_or_400(
        ledger.record_fit_aggregation, request.node_id,
        request.server_round, request.update_hash,
    )


def call_or_400(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/server-key")
def server_key() -> dict:
    return {"algorithm": "Ed25519", "public_key": ledger.server_public_key}


@app.post("/nodes/register")
def register(request: Registration) -> dict:
    return call_or_400(
        ledger.register_node,
        request.node_id,
        request.profile,
        request.public_key,
        request.registration_signature,
    )


@app.post("/nodes/{node_id}/tasks")
def issue_task(node_id: str) -> dict:
    return call_or_400(ledger.issue_task, node_id)


@app.post("/tasks/receipt")
def acknowledge(request: Receipt) -> dict:
    return call_or_400(
        ledger.acknowledge,
        request.node_id,
        request.task_id,
        request.assignment_hash,
        request.signature,
    )


@app.post("/tasks/result")
def result(request: Result) -> dict:
    return call_or_400(
        ledger.submit_result,
        request.node_id,
        request.task_id,
        request.quality,
        request.result_hash,
        request.signature,
        request.work_product,
        shadow_update=request.shadow_update,
    )


@app.post("/flower/events")
def flower_event(request: FlowerEvent) -> dict:
    return call_or_400(
        ledger.record_flower_event,
        request.node_id,
        request.event_id,
        request.event_type,
        request.metrics_hash,
        request.signature,
    )


@app.post("/experiment/onboarding-finished")
def onboarding_finished(request: OnboardingFinished) -> dict:
    return call_or_400(
        ledger.mark_onboarding_finished,
        request.node_id,
        request.final_access_state,
        request.signature,
    )


@app.get("/experiment/barrier")
def experiment_barrier() -> dict:
    return ledger.barrier_status()


@app.get("/nodes/{node_id}/status")
def status(node_id: str) -> dict:
    return call_or_400(ledger.node_status, node_id)


@app.get("/nodes/{node_id}/records")
def records(node_id: str) -> list[dict]:
    return call_or_400(ledger.task_records, node_id)


@app.get("/audit/verify")
def audit_verify() -> dict:
    return ledger.verify_integrity()


@app.get("/experiment/summary")
def experiment_summary() -> dict:
    return ledger.experiment_summary()
