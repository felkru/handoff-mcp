"""handoff-mcp: hand tasks off to peer coding-agent CLIs (Codex / Antigravity)."""

from handoff_mcp.core import (
    BACKENDS,
    HandoffError,
    backends_status,
    get_job,
    list_jobs,
    run_backend,
    start_job,
)

__all__ = [
    "run_backend",
    "start_job",
    "get_job",
    "list_jobs",
    "backends_status",
    "BACKENDS",
    "HandoffError",
]
__version__ = "0.1.0"
