"""`handoff` CLI — same engine as the MCP server, for shell/scripts.

Examples:
    handoff codex "review this for bugs" -f src/core.py
    handoff antigravity "what's wrong with this UI?" -f screenshot.png
    handoff codex "refactor to async" -f app.py --approve
    handoff --status
    handoff --guidance
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from handoff_mcp import core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="handoff",
        description="Hand a task off to a peer coding-agent CLI (codex / "
        "antigravity). Separate quota pools; independent second opinions.",
    )
    p.add_argument("backend", nargs="?", choices=list(core.BACKENDS),
                   help="Which peer agent to use.")
    p.add_argument("prompt", nargs="?", help="Prompt for the peer agent.")
    p.add_argument("-f", "--file", action="append", default=[], dest="files",
                   help="Attach a file (repeatable).")
    p.add_argument("-m", "--model", default="", help="Model id override.")
    p.add_argument("-C", "--cwd", default=None,
                   help="Project dir the agent may read for context.")
    p.add_argument("--approve", action="store_true",
                   help="Allow the agent to make edits (default: read-only).")
    p.add_argument("--timeout", type=int, default=core.DEFAULT_TIMEOUT,
                   help="Seconds before abort.")
    p.add_argument("--async", action="store_true", dest="is_async",
                   help="Run as a background job, tailing status until done.")
    p.add_argument("--status", action="store_true",
                   help="Print backend install/auth status and exit.")
    p.add_argument("--guidance", action="store_true",
                   help="Print the model routing guide and exit.")
    return p


def _run_async(args) -> int:
    job_id = core.start_job(
        args.backend, args.prompt, files=args.files, model=args.model,
        cwd=args.cwd, approve=args.approve,
    )
    print(f"job {job_id} ({args.backend}) started", file=sys.stderr)
    seen = 0
    while True:
        job = core.get_job(job_id, tail=200)
        lines = job["log_tail"].splitlines()
        for line in lines[seen:]:
            print(line, file=sys.stderr)
        seen = len(lines)
        if job["state"] != "running":
            if job["state"] == "done":
                print(job["result"])
                return 0
            print(f"error: {job['error']}", file=sys.stderr)
            return 1
        time.sleep(2)


def main() -> int:
    args = build_parser().parse_args()

    if args.status:
        print(json.dumps(core.backends_status(), indent=2))
        return 0
    if args.guidance:
        print(core.staleness_banner() + "\n\n" + core.ROUTING_GUIDE)
        return 0

    if not args.backend or not args.prompt:
        print("error: need BACKEND and PROMPT (or --status / --guidance).\n"
              f"backends: {', '.join(core.BACKENDS)}", file=sys.stderr)
        return 2

    try:
        if args.is_async:
            return _run_async(args)
        result = core.run_backend(
            args.backend, args.prompt, files=args.files, model=args.model,
            cwd=args.cwd, approve=args.approve, timeout=args.timeout,
        )
    except core.HandoffError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
