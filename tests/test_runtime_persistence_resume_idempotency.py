from __future__ import annotations

import asyncio

import pytest

from ksadk.conversations.runtime_persistence import append_run_resume_event
from ksadk.sessions.in_memory import InMemorySessionService
from ksadk.sessions.local_service import LocalSessionService

pytestmark = pytest.mark.asyncio


async def _append(service, *, invocation_id: str, checkpoint_id: str = "checkpoint-1"):
    return await append_run_resume_event(
        session_id="session-1",
        author="agent-1",
        run_id="run-1",
        checkpoint_id=checkpoint_id,
        resume_attempt_id="attempt-1",
        framework="langgraph",
        framework_ref={"langgraph": {"checkpoint_id": checkpoint_id}},
        invocation_id=invocation_id,
        session_service_provider=lambda: service,
    )


@pytest.mark.parametrize("backend", ["memory", "local"])
async def test_resume_attempt_is_idempotent_across_invocation_retries(tmp_path, backend):
    service = (
        InMemorySessionService()
        if backend == "memory"
        else LocalSessionService(tmp_path / "sessions.sqlite")
    )
    await service.create_session("agent-1", "user-1", session_id="session-1")

    first = await _append(service, invocation_id="invocation-a")
    retry = await _append(service, invocation_id="invocation-b")

    assert retry.id == first.id
    assert retry.invocation_id == "invocation-a"
    events = await service.get_events("session-1")
    assert [event.event_type for event in events] == ["run_resume"]


async def test_resume_attempt_collision_fails_loudly(tmp_path):
    service = InMemorySessionService()
    await service.create_session("agent-1", "user-1", session_id="session-1")
    await _append(service, invocation_id="invocation-a")

    with pytest.raises(ValueError, match="already bound to a different checkpoint"):
        await _append(service, invocation_id="invocation-b", checkpoint_id="checkpoint-2")


async def test_concurrent_resume_retries_share_one_deterministic_event():
    service = InMemorySessionService()
    await service.create_session("agent-1", "user-1", session_id="session-1")

    results = await asyncio.gather(
        _append(service, invocation_id="invocation-a"),
        _append(service, invocation_id="invocation-b"),
    )

    assert results[0].id == results[1].id
    events = await service.get_events("session-1")
    assert len(events) == 1
    assert events[0].metadata["resume_attempt_id"] == "attempt-1"
