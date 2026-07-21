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
  * handoff          — run a backend synchronously, get the answer back.
  * handoff_start    — fire a long job, get a job id immediately (async).
  * handoff_status   — poll a job: state, elapsed, streamed progress, result.
  * handoff_jobs     — list recent jobs.
  * handoff_backends — install/auth readiness of each backend + staleness note.
  * handoff_guidance — the model routing guide (who's good at what).

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

    Use this for tasks that finish in a few minutes. For long/agentic work,
    prefer `handoff_start` + `handoff_status` so you get progress and don't block.

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

    Args mirror `handoff`. Returns a short job id string (pass it to
    `handoff_status`). The job's own timeout is generous (default 30 min).
    """
    try:
        job_id = core.start_job(
            backend, prompt, files=files, model=model, cwd=cwd, approve=approve,
        )
    except HandoffError as e:
        return f"[handoff error] {e}"
    return (
        f"Started {backend} job {job_id}. Poll it with "
        f"handoff_status(job_id='{job_id}')."
    )


@mcp.tool()
def handoff_status(job_id: str, tail: int = 40) -> str:
    """Poll a background handoff job: state, elapsed time, live progress, result.

    Args:
        job_id: The id returned by `handoff_start`.
        tail: How many lines of streamed progress log to include.

    Returns JSON with: `state` (running|done|error), `elapsed_s`, `backend`,
    `log_tail` (the peer agent's streamed activity so far), and — once done —
    `result` (or `error`). Call repeatedly until `state` != "running".
    """
    try:
        return json.dumps(core.get_job(job_id, tail=tail), indent=2)
    except HandoffError as e:
        return f"[handoff error] {e}"


@mcp.tool()
def handoff_jobs(limit: int = 10) -> str:
    """List recent handoff jobs (id, backend, state, elapsed, prompt preview)."""
    return json.dumps(core.list_jobs(limit=limit), indent=2)


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
