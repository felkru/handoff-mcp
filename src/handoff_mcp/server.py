"""FastMCP server: hand a task off to a peer coding-agent CLI (Codex / Antigravity).

Why hand off?
  1. A genuinely independent model — Codex (OpenAI) or Gemini 3 (Google) — gives
     a real second/adversarial opinion instead of Claude grading its own work.
  2. Each backend runs on its OWN subscription quota, so offloading work
     preserves this session's Claude (Fable) usage limit at no extra cost.
  3. Visual feedback: Antigravity/Gemini 3 is natively multimodal.

>>> BE AGGRESSIVE ABOUT HANDING OFF WHEN THE FABLE USAGE LIMIT IS NEAR. <<<
When the harness signals you are close to the Claude/Fable usage limit (or a
long, expensive task is coming), prefer delegating suitable work to a backend
rather than spending Fable budget — bulk review, long reads, second opinions,
and mechanical agentic coding are all good candidates. See `handoff_guidance`
for full routing.

Tools:
  * handoff             — run a backend synchronously, get the answer back.
  * handoff_start       — fire a long job, get a job id immediately (async).
  * handoff_wait        — BLOCK until a job finishes and return its result.
  * handoff_status      — peek at a job mid-flight: state, elapsed, live progress.
  * handoff_map         — fan MANY tasks out at once (parallel, one group handle).
  * handoff_group_wait  — BLOCK until a whole fan-out group finishes.
  * handoff_group_status— progress peek at a fan-out group.
  * handoff_jobs        — list recent jobs (memory + disk, survives restarts).
  * handoff_backends    — install/auth readiness of each backend + staleness note.
  * handoff_guidance    — the model routing guide (who's good at what).

REACHING NATIVE-SUBAGENT PARITY — pick the shape that matches the work
---------------------------------------------------------------------
  * one quick delegation          → `handoff` (blocking, minutes)
  * one long delegation           → `handoff_start` + `handoff_wait`
  * N independent delegations     → `handoff_map` + `handoff_group_wait`
Reach for `handoff_map` in exactly the situations where you would otherwise
spawn several native subagents in one message. Sending the *same* artefact to
both backends is a first-class use: two independent models disagreeing is the
main thing this server buys you that a Claude subagent cannot.

WHEN QUOTA RUNS OUT — YOU MUST FALL BACK TO A NATIVE SUBAGENT
-------------------------------------------------------------
Backends bill to separate pools, so an exhausted quota on one says nothing about
the other: every call rotates to the sibling backend automatically before giving
up (skipping it when it cannot handle the attachments, e.g. PDFs on codex).

If BOTH are exhausted you get a result whose text begins with

    [handoff exhausted → use native Agent]

That string is a binding instruction, not an error to report: immediately re-run
the task with the native `Agent` tool and carry on. Do NOT retry handoff, do not
wait for quota, and do not abandon the task. This server cannot call `Agent`
itself, so this handover only happens if you act on the marker.

COLLECTING AN ASYNC JOB — read this before using `handoff_start`
----------------------------------------------------------------
`handoff_start` returns immediately, so *something* has to bring you back when
the job finishes. Three supported ways, in order of preference:

  1. `handoff_wait(job_id)` — blocks server-side and returns the result the
     instant the job is done. Simplest and never misses the completion.
  2. Background shell waiter on the sentinel file, when you want to keep
     working meanwhile:
         until [ -f ~/.cache/handoff-mcp/jobs/<id>.done ]; do sleep 10; done
     `handoff_start` prints the exact sentinel path; `handoff_status` returns it
     as `done_sentinel`. The file is created only at a terminal state, so the
     command exits exactly once — which is what makes an agent harness fire its
     completion notification.
  3. `handoff_status(job_id)` in a loop — manual polling; use only for progress.

Anti-pattern (this cost a real session its wakeup): backgrounding a loop that
waits on a file the job never creates. Jobs write `<id>.log` (live output),
`<id>.json` (full state, survives an MCP restart) and `<id>.done` (terminal
sentinel) into ~/.cache/handoff-mcp/jobs/ — nothing else. A waiter watching any
other path spins until it is killed and silently never notifies anyone.

All logic lives in `handoff_mcp.core`; these are thin wrappers.
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from handoff_mcp import core
from handoff_mcp.core import HandoffError

log = logging.getLogger("handoff-mcp")

mcp = FastMCP("handoff-mcp")


@mcp.tool()
def handoff(
    prompt: str,
    backend: str,
    files: list[str] | None = None,
    model: str = "",
    cwd: str | None = None,
    approve: bool = False,
    timeout: int = 600,
) -> str:
    """Hand a task off to a peer coding-agent CLI and return its answer (blocking).

    Use this for ONE task that finishes in a few minutes. For long/agentic work
    prefer `handoff_start` + `handoff_wait`; for 2+ independent tasks use
    `handoff_map`, which runs them in parallel instead of serialising them here.

    Falls back to the sibling backend automatically if this one is out of quota.
    If the returned text starts with "[handoff exhausted → use native Agent]",
    both pools are empty: re-run this task with the native `Agent` tool now.

    Choosing a backend (see `handoff_guidance` for detail):
      - `codex` (OpenAI): code review, bug hunting, refactors, terminal-agentic
        work, second opinion on code/text. Text & images (not PDFs).
      - `antigravity` (Google Gemini 3): visual feedback (screenshots, diagrams,
        PDFs), very-long-context reads (whole repos / huge docs), cheap bulk.

    Hand off aggressively when the Claude/Fable usage limit is close — each
    backend bills to a separate subscription pool.

    Args:
        prompt: What to do. The peer agent has NONE of this conversation's
            context — be explicit and self-contained.
        backend: "codex" or "antigravity".
        files: Absolute/~ paths to attach. codex: text + images. antigravity:
            text, images, and PDFs.
        model: Optional model id override; empty = the CLI's own default
            (e.g. "gpt-5.3-codex", "gemini-3.1-pro-preview").
        cwd: A project directory the agent may read for extra context. The
            process itself runs in a throwaway temp dir; originals are never
            modified unless you set `approve=True`.
        approve: Allow the agent to actually make edits (codex: workspace-write
            sandbox; antigravity: --dangerously-skip-permissions). Default False
            = read-only, right for reviews/opinions.
        timeout: Seconds before abort.

    Returns:
        The peer agent's response, or a clear "[handoff error] …" string.
    """
    try:
        return core.run_backend(
            backend, prompt, files=files, model=model, cwd=cwd,
            approve=approve, timeout=timeout,
        )
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_start(
    prompt: str,
    backend: str,
    files: list[str] | None = None,
    model: str = "",
    cwd: str | None = None,
    approve: bool = False,
) -> str:
    """Start a handoff as a background job and return a job id immediately.

    Use this for long or agentic handoffs (deep reviews, multi-file refactors,
    reading a huge document). Then poll `handoff_status(job_id)` to watch
    progress stream in and collect the result when it's `done`. This is the
    async path — it lets you keep working / check in periodically instead of
    blocking on one long call.

    Args mirror `handoff`. Returns a short job id string. The job's own timeout
    is generous (default 30 min).

    To collect the result, prefer `handoff_wait(job_id)` — it blocks until the
    job is finished and returns it, so you get woken up instead of guessing a
    polling cadence. Use `handoff_status(job_id)` only when you actually want a
    mid-flight progress peek. If you would rather keep working in the meantime,
    wait on the sentinel file in a background shell command
    (`until [ -f <done_sentinel> ]; do sleep 10; done`) — that command finishes
    when the job does, which is what makes the harness notify you. Never poll a
    file the job does not create: a loop that never exits never notifies.
    """
    try:
        job_id = core.start_job(
            backend, prompt, files=files, model=model, cwd=cwd, approve=approve,
        )
    except HandoffError as e:
        return f"[handoff error] {e}"
    return (
        f"Started {backend} job {job_id}.\n"
        f"  Collect it:  handoff_wait(job_id='{job_id}')   ← blocks until done\n"
        f"  Peek at it:  handoff_status(job_id='{job_id}')\n"
        f"  Sentinel:    {core.sentinel_for(job_id)}  (appears when finished)"
    )


@mcp.tool()
def handoff_wait(job_id: str, timeout: float = 1800.0, tail: int = 40) -> str:
    """Block until a background handoff job finishes, then return its result.

    This is the notification path for `handoff_start`: the tool call returns the
    moment the job reaches `done`/`error`, so you never have to pick a polling
    interval or risk missing the completion.

    Args:
        job_id: The id returned by `handoff_start`.
        timeout: Max seconds to block. On expiry you get the still-running
            status with `wait_timed_out: true` — call again to keep waiting.
        tail: Lines of streamed progress log to include.

    Returns the same JSON as `handoff_status`, with `result` populated.
    """
    try:
        return json.dumps(core.wait_job(job_id, timeout=timeout, tail=tail), indent=2)
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_status(job_id: str, tail: int = 40) -> str:
    """Poll a background handoff job: state, elapsed time, live progress, result.

    Args:
        job_id: The id returned by `handoff_start`.
        tail: How many lines of streamed progress log to include.

    Returns JSON with: `state` (running|done|error|orphaned), `elapsed_s`,
    `backend`, `log_tail` (the peer agent's streamed activity so far),
    `done_sentinel` (file that appears when the job ends — wait on this from a
    background shell if you want to be notified), and — once done — `result`
    (or `error`).

    Prefer `handoff_wait(job_id)` to collect a result; it returns as soon as the
    job is finished. Use this tool for a progress peek. `state: "orphaned"`
    means the MCP server restarted mid-job, so it will never complete — re-run it.
    """
    try:
        return json.dumps(core.get_job(job_id, tail=tail), indent=2)
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_jobs(limit: int = 10) -> str:
    """List recent handoff jobs (id, backend, group, state, elapsed, prompt preview).

    Merges the live registry with the on-disk mirror, so jobs started before an
    MCP restart still appear (as `orphaned` if no worker survived).
    """
    return json.dumps(core.list_jobs(limit=limit), indent=2)


@mcp.tool()
def handoff_map(
    tasks: list[dict],
    backend: str = "codex",
    cwd: str | None = None,
    approve: bool = False,
) -> str:
    """Fan several tasks out to peer agents CONCURRENTLY and return one group handle.

    This is the parity path for spawning multiple native subagents at once: use
    it whenever you have 2+ independent delegable units (review N files, read N
    documents, get N opinions on the same artefact). All tasks start immediately
    and run in parallel; wall-clock is the slowest task, not the sum.

    Args:
        tasks: One dict per task. `prompt` is required; optional per-task keys
            are `backend`, `files`, `model`, `label` (a short tag that keeps the
            result set readable). Each prompt must be fully self-contained — peer
            agents share none of this conversation's context and none of each
            other's.
        backend: Default backend for tasks that don't name their own. Mixing is
            useful and encouraged: send the same artefact to both `codex` and
            `antigravity` for genuinely independent opinions.
        cwd: Default project directory tasks may read.
        approve: Allow edits (default False = read-only, right for reviews).

    Returns JSON with `group_id`, the per-task `jobs` (each with its own
    `job_id`), and `group_sentinel` — a single file that appears once ALL tasks
    are terminal. Collect with `handoff_group_wait(group_id)`, or keep working
    and wait on the group sentinel from one background shell command:
        until [ -f <group_sentinel> ]; do sleep 10; done
    One waiter for the whole batch; do not start one per job.
    """
    try:
        return json.dumps(
            core.start_group(tasks, backend=backend, cwd=cwd, approve=approve),
            indent=2,
        )
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_group_wait(group_id: str, timeout: float = 1800.0, tail: int = 10) -> str:
    """Block until every task in a `handoff_map` group finishes; return all results.

    The collection path for fan-out, analogous to awaiting a batch of native
    subagents. On timeout you get partial results with `wait_timed_out: true`;
    call again to keep waiting.
    """
    try:
        return json.dumps(core.wait_group(group_id, timeout=timeout, tail=tail), indent=2)
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_group_status(group_id: str, tail: int = 10) -> str:
    """Progress peek at a `handoff_map` group: aggregate state plus every task.

    Use `handoff_group_wait` to actually collect; this is for mid-flight checks.
    """
    try:
        return json.dumps(core.get_group(group_id, tail=tail), indent=2)
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_backends() -> str:
    """Report each backend's install + auth readiness, its quota pool, and a
    freshness note for the routing guidance.

    Call this first if a handoff returns an auth/not-found error. The output
    includes a `staleness` banner reminding the maintainer to refresh the model
    guidance when new models ship.
    """
    return json.dumps(core.backends_status(), indent=2)


@mcp.tool()
def handoff_guidance() -> str:
    """Return the model routing guide: which backend is best for which task, and
    when to hand off aggressively to preserve the Claude/Fable usage limit.

    NOTE TO MAINTAINER: this reflects the model landscape as of
    core.MODEL_INFO_UPDATED. Update core.ROUTING_GUIDE when new models/CLIs ship.
    """
    return core.staleness_banner() + "\n\n" + core.ROUTING_GUIDE


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
