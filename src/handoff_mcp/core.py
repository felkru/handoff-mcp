"""Engine for handing a task off to a peer coding-agent CLI.

Two backends, each on its own subscription quota (so running one never eats the
other's limit):

* ``codex``       — OpenAI's ``codex exec`` (ChatGPT login). Strong at code
                    review / reasoning. Proper headless mode with JSONL
                    streaming and a clean final-message file.
* ``antigravity`` — Google's ``agy`` (Google AI Pro login), the successor to
                    the retired ``gemini`` CLI. Gemini 3 models: multimodal
                    visual feedback and very-long-context review.

Everything the MCP server and the ``handoff`` CLI expose is a thin wrapper over
this module (DRY). Two execution primitives:

* Codex runs as an ordinary subprocess — it is headless-native.
* ``agy`` gates its stdout on ``isatty()`` (upstream bug #76: empty output in a
  non-TTY). We therefore run it inside a pseudo-terminal and strip the ANSI /
  spinner noise back out.

A small in-process job registry lets callers fire a long handoff and poll it
(:func:`start_job` / :func:`get_job`) instead of blocking, so the caller can
receive incremental status while the peer agent works.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# ── configuration ──────────────────────────────────────────────────────────

BACKENDS = ("codex", "antigravity")

# Binaries (override via env for non-standard installs).
CODEX_BIN = os.environ.get("HANDOFF_CODEX_BIN", "codex")
AGY_BIN = os.environ.get("HANDOFF_AGY_BIN", "agy")

DEFAULT_TIMEOUT = int(os.environ.get("HANDOFF_TIMEOUT", "600"))
JOB_TIMEOUT = int(os.environ.get("HANDOFF_JOB_TIMEOUT", "1800"))

# Where job logs live so status survives across tool calls / restarts.
HOME_DIR = Path(os.environ.get("HANDOFF_MCP_HOME", Path.home() / ".cache" / "handoff-mcp"))
JOBS_DIR = HOME_DIR / "jobs"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".tiff"}

# Model left empty → each CLI uses its own configured default.
DEFAULT_MODEL = os.environ.get("HANDOFF_MODEL", "")

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class HandoffError(RuntimeError):
    """Raised when a backend is missing, unauthenticated, or fails."""


# ── model landscape (MAINTAINER: refresh when new models/CLIs ship) ──────────

# The date this routing guidance was last verified against the model landscape.
# `backends_status()` surfaces a staleness banner built from it, and the CLI /
# server repeat the reminder, so the guidance never silently rots.
MODEL_INFO_UPDATED = "2026-07-21"

# How stale the guidance may get before the banner escalates from a note to a
# warning (days).
_STALE_AFTER_DAYS = 120

# Concise, opinionated routing guide baked into the tool docs so the calling
# model knows *when* to hand off and *to which* backend. Rewrite this whenever
# a new model or CLI meaningfully changes the picture.
ROUTING_GUIDE = f"""\
Handoff routing guide (verified {MODEL_INFO_UPDATED}; MAINTAINER: update when
new models/CLIs ship):

BE AGGRESSIVE ABOUT HANDING OFF. This session runs on the user's Claude (Fable)
subscription, which has a finite usage limit. When that limit is getting close
(the harness warns you, or a long/expensive task is coming up), prefer handing
work to a backend below instead of spending Fable budget — especially for work
another model does at least as well (bulk review, long reads, second opinions,
mechanical agentic coding). Each backend runs on a SEPARATE subscription pool,
so a handoff conserves Fable quota at no extra cost to the user.

Pick the backend by task:

• codex  (OpenAI GPT-5.x-Codex, via ChatGPT login → OpenAI quota)
    Best at: agentic / terminal / CLI coding (Terminal-Bench ~77% vs Gemini
    ~54%), long-horizon autonomous work, large refactors & migrations,
    adversarial CODE review and bug hunting, cybersecurity, long-context
    RELIABILITY (holds instructions over big inputs).
    Weak at: images / diagrams / UI mockups (text-specialized); the Codex
    surface caps context ~400K.
    Reach for it: "review this diff/codebase", "hunt for the bug", "refactor X",
    "is this argument sound" when the artifact is code/text.

• antigravity  (Google Gemini 3 Pro via `agy`, Google AI Pro login → Google
    quota; successor to the retired `gemini` CLI)
    Best at: VISUAL feedback (screenshots, diagrams, charts, UI, PDFs — natively
    multimodal), VERY long context (1M–2M tokens: whole monorepos / long docs),
    novel reasoning puzzles, cost-sensitive bulk work.
    Weak at: terminal-agentic coding; the `agy` CLI is finicky headless (we run
    it under a PTY to work around its isatty output bug).
    Reach for it: "what's wrong with this screenshot/design", "read this 400-page
    PDF and …", "analyze this entire repo at once".

Rule of thumb: code/agentic/second-opinion-on-text → codex. Anything visual, or
anything too large for a normal context window → antigravity. When near the
Fable limit, lower the bar for handing off in either direction.
"""


def staleness_banner() -> str:
    """A one-line freshness notice; escalates to a warning past the threshold.

    Kept deliberately simple (no wall-clock math beyond a date diff) so it is
    cheap to compute on every status call.
    """
    try:
        updated = time.strptime(MODEL_INFO_UPDATED, "%Y-%m-%d")
        age_days = (time.time() - time.mktime(updated)) / 86400
    except (ValueError, OverflowError):
        return f"Model routing guidance dated {MODEL_INFO_UPDATED}."
    if age_days > _STALE_AFTER_DAYS:
        return (
            f"⚠️  MAINTAINER ACTION: routing guidance is {int(age_days)} days old "
            f"(since {MODEL_INFO_UPDATED}). New Gemini/GPT/Codex versions have "
            f"likely shipped — review ROUTING_GUIDE and MODEL_INFO_UPDATED in "
            f"handoff_mcp/core.py and refresh the strengths/weaknesses."
        )
    return (
        f"Model routing guidance verified {MODEL_INFO_UPDATED} "
        f"({int(age_days)}d ago). Update ROUTING_GUIDE in core.py when new "
        f"models/CLIs appear."
    )


# ── helpers ────────────────────────────────────────────────────────────────

def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _safe_name(index: int, path: Path) -> str:
    base = _SAFE.sub("_", path.name).strip("_") or "file"
    return f"{index:02d}_{base}"


def _strip_terminal(raw: str) -> str:
    """Turn raw PTY output into plain text: drop ANSI, collapse \\r repaints."""
    text = _ANSI.sub("", raw)
    # Normalize CRLF line endings first; only *bare* \r is a spinner repaint,
    # where we keep the last write on the line.
    text = text.replace("\r\n", "\n")
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    keep = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and all(ch in "─│╭╮╰╯━┃▝▜▄▀▖▗▘▙▚▛▜▟● " for ch in stripped):
            continue
        keep.append(line.rstrip())
    return "\n".join(keep).strip()


def _classify(files: list[str]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split attachments into (images, text, other) resolved Paths."""
    images, text, other = [], [], []
    for raw in files:
        p = Path(raw).expanduser()
        if not p.exists():
            raise HandoffError(f"Attached file does not exist: {raw}")
        p = p.resolve()
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext == ".pdf":
            other.append(p)
        else:
            try:
                p.read_text(encoding="utf-8")
                text.append(p)
            except (UnicodeDecodeError, OSError):
                other.append(p)
    return images, text, other


def _inline_text_files(prompt: str, text_files: list[Path]) -> str:
    if not text_files:
        return prompt
    blocks = [prompt, "\n\n--- Attached files ---"]
    for p in text_files:
        try:
            content = p.read_text(encoding="utf-8")
        except OSError as e:
            raise HandoffError(f"Could not read {p}: {e}")
        blocks.append(f"\n### {p.name}\n```\n{content}\n```")
    return "\n".join(blocks)


# ── codex backend (subprocess, headless-native) ─────────────────────────────

def _run_codex(prompt, *, files, model, cwd, timeout, log_path, approve) -> str:
    binary = _which(CODEX_BIN)
    if not binary:
        raise HandoffError(
            "codex CLI not found. Install it (`npm i -g @openai/codex` or "
            "`brew install codex`) and run `codex login`."
        )
    images, text_files, other = _classify(files)
    if other:
        names = ", ".join(p.name for p in other)
        raise HandoffError(
            f"codex can't attach {names} (non-image binary/PDF). Use "
            "backend='antigravity' for PDFs / complex visual input."
        )
    full_prompt = _inline_text_files(prompt, text_files)

    with tempfile.TemporaryDirectory(prefix="handoff-codex-") as tmp:
        last_msg = Path(tmp) / "last.txt"
        sandbox = "workspace-write" if approve else "read-only"
        argv = [binary, "exec", "--json", "--skip-git-repo-check",
                "-s", sandbox, "-o", str(last_msg)]
        if model:
            argv += ["-m", model]
        if cwd:
            argv += ["-C", str(Path(cwd).expanduser().resolve())]
        for img in images:
            argv += ["-i", str(img)]
        # Pass the prompt via stdin, NOT as a positional: codex's `-i/--image`
        # flag is variadic and would otherwise swallow a trailing prompt arg.
        rc = _stream_subprocess(argv, cwd=tmp, timeout=timeout,
                                log_path=log_path, stdin_data=full_prompt)
        final = last_msg.read_text(encoding="utf-8").strip() if last_msg.exists() else ""
        if not final:
            tail = _tail_file(log_path, 30) if log_path else ""
            if rc != 0:
                raise HandoffError(f"codex exited {rc}. Last output:\n{tail}")
            raise HandoffError(f"codex produced no final message. Log tail:\n{tail}")
        return final


# ── antigravity backend (agy via PTY, bug #76 workaround) ────────────────────

def _run_antigravity(prompt, *, files, model, cwd, timeout, log_path, approve) -> str:
    binary = _which(AGY_BIN)
    if not binary:
        raise HandoffError(
            "agy (Antigravity CLI) not found. Install with "
            "`brew install --cask antigravity-cli` and run `agy` once to log in."
        )
    with tempfile.TemporaryDirectory(prefix="handoff-agy-") as tmp:
        workspace = Path(tmp)
        refs = []
        for i, raw in enumerate(files):
            src = Path(raw).expanduser().resolve()
            if not src.exists():
                raise HandoffError(f"Attached file does not exist: {raw}")
            name = _safe_name(i, src)
            shutil.copy2(src, workspace / name)  # copy, never touch originals
            refs.append(f"@{name}")
        full_prompt = ("\n".join(refs) + "\n\n" + prompt) if refs else prompt

        # agy headless can't prompt for tool permissions, so it auto-DENIES —
        # which blocks even reading an attached @file. Auto-approve is safe here
        # because the process runs in a throwaway temp workspace holding only
        # *copies*. Edits are gated by mode: `plan` = strictly read-only (right
        # for reviews/opinions), `accept-edits` only when the caller opts in.
        argv = [binary, "-p", full_prompt, "--dangerously-skip-permissions",
                "--mode", "accept-edits" if approve else "plan"]
        if model:
            argv += ["--model", model]
        if cwd:
            argv += ["--add-dir", str(Path(cwd).expanduser().resolve())]

        raw = _stream_pty(argv, cwd=str(workspace), timeout=timeout, log_path=log_path)
        text = _strip_terminal(raw)
        if not text:
            raise HandoffError(
                "agy returned no output. Ensure you have logged in (`agy` once) "
                "and that your Google AI Pro quota is not exhausted."
            )
        return text


# ── process primitives ───────────────────────────────────────────────────────

def _stream_subprocess(argv, *, cwd, timeout, log_path, stdin_data=None) -> int:
    """Run a normal subprocess, streaming stdout+stderr into log_path live.

    If ``stdin_data`` is given it is written to the child's stdin and closed
    (used to pass codex its prompt without a swallowable positional arg).
    """
    proc = subprocess.Popen(
        argv, cwd=cwd, env=os.environ.copy(),
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    if stdin_data is not None and proc.stdin is not None:
        proc.stdin.write(stdin_data)
        proc.stdin.close()
    start = time.time()
    fh = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if fh:
                fh.write(_codex_event_line(line))
                fh.flush()
            if time.time() - start > timeout:
                proc.kill()
                raise HandoffError(f"codex timed out after {timeout}s.")
        return proc.wait(timeout=30)
    finally:
        if fh:
            fh.close()


def _codex_event_line(line: str) -> str:
    """Render a codex --json event as a short human status line for the log."""
    line = line.strip()
    if not line:
        return ""
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return line + "\n"
    t = ev.get("type") or ev.get("msg", {}).get("type", "")
    item = ev.get("item", ev)
    kind = item.get("type", "") if isinstance(item, dict) else ""
    if kind == "command_execution":
        return f"$ {str(item.get('command',''))[:120]}\n"
    if kind == "reasoning":
        return f"[reasoning] {str(item.get('text',''))[:120]}\n"
    if kind:
        return f"[{kind}]\n"
    return f"[{t}]\n" if t else ""


def _stream_pty(argv, *, cwd, timeout, log_path) -> str:
    """Run a command inside a PTY so isatty()-gated CLIs (agy) emit output."""
    import pty
    import select

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        argv, cwd=cwd, env=os.environ.copy(),
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
    )
    os.close(slave)
    chunks: list[str] = []
    start = time.time()
    fh = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        while True:
            if time.time() - start > timeout:
                proc.kill()
                raise HandoffError(f"agy timed out after {timeout}s.")
            r, _, _ = select.select([master], [], [], 0.5)
            if master in r:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if not data:
                    break
                text = data.decode("utf-8", errors="replace")
                chunks.append(text)
                if fh:
                    fh.write(_strip_terminal(text) + "\n")
                    fh.flush()
            elif proc.poll() is not None:
                break
        proc.wait(timeout=30)
    finally:
        os.close(master)
        if fh:
            fh.close()
    return "".join(chunks)


def _tail_file(path, n: int) -> str:
    path = Path(path) if path else None
    if not path or not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


# ── dispatch ─────────────────────────────────────────────────────────────────

_RUNNERS = {"codex": _run_codex, "antigravity": _run_antigravity}


def run_backend(backend, prompt, *, files=None, model=DEFAULT_MODEL, cwd=None,
                timeout=DEFAULT_TIMEOUT, approve=False, log_path=None) -> str:
    if backend not in _RUNNERS:
        raise HandoffError(f"Unknown backend {backend!r}. Choose one of {BACKENDS}.")
    if not prompt or not prompt.strip():
        raise HandoffError("prompt must be non-empty")
    return _RUNNERS[backend](
        prompt, files=files or [], model=model, cwd=cwd,
        timeout=timeout, log_path=log_path, approve=approve,
    )


# ── job registry (async status) ──────────────────────────────────────────────

@dataclass
class Job:
    id: str
    backend: str
    prompt: str
    state: str = "running"  # running | done | error
    started: float = 0.0
    finished: float | None = None
    result: str | None = None
    error: str | None = None
    log_path: str = ""

    def public(self, tail: int = 40) -> dict:
        d = asdict(self)
        elapsed = (self.finished or time.time()) - self.started
        d["elapsed_s"] = round(elapsed, 1)
        d["log_tail"] = _tail_file(self.log_path, tail)
        if self.state == "running":
            d["result"] = None
        return d


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _worker(job: Job, kwargs: dict) -> None:
    try:
        result = run_backend(job.backend, log_path=Path(job.log_path), **kwargs)
        with _LOCK:
            job.result, job.state, job.finished = result, "done", time.time()
    except Exception as e:  # noqa: BLE001 - surface any failure to the poller
        with _LOCK:
            job.error, job.state, job.finished = str(e), "error", time.time()


def start_job(backend, prompt, *, files=None, model=DEFAULT_MODEL, cwd=None,
              timeout=JOB_TIMEOUT, approve=False) -> str:
    if backend not in _RUNNERS:
        raise HandoffError(f"Unknown backend {backend!r}. Choose one of {BACKENDS}.")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    log_path = JOBS_DIR / f"{job_id}.log"
    log_path.write_text("", encoding="utf-8")
    job = Job(id=job_id, backend=backend, prompt=prompt[:200],
              started=time.time(), log_path=str(log_path))
    with _LOCK:
        _JOBS[job_id] = job
    kwargs = dict(prompt=prompt, files=files, model=model, cwd=cwd,
                  timeout=timeout, approve=approve)
    threading.Thread(target=_worker, args=(job, kwargs), daemon=True).start()
    return job_id


def get_job(job_id: str, tail: int = 40) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HandoffError(f"No such job {job_id!r}. It may predate a restart.")
    return job.public(tail=tail)


def list_jobs(limit: int = 10) -> list[dict]:
    with _LOCK:
        jobs = sorted(_JOBS.values(), key=lambda j: j.started, reverse=True)[:limit]
    return [
        {"id": j.id, "backend": j.backend, "state": j.state,
         "elapsed_s": round((j.finished or time.time()) - j.started, 1),
         "prompt": j.prompt[:80]}
        for j in jobs
    ]


# ── auth / readiness ─────────────────────────────────────────────────────────

def backends_status() -> dict:
    out: dict = {
        "guidance_updated": MODEL_INFO_UPDATED,
        "staleness": staleness_banner(),
    }

    codex_path = _which(CODEX_BIN)
    codex: dict = {"installed": bool(codex_path), "path": codex_path,
                   "quota": "OpenAI / ChatGPT subscription"}
    if codex_path:
        try:
            v = subprocess.run([codex_path, "--version"], capture_output=True,
                               text=True, timeout=15)
            codex["version"] = (v.stdout or v.stderr).strip()
            s = subprocess.run([codex_path, "login", "status"], capture_output=True,
                               text=True, timeout=15)
            line = (s.stdout + s.stderr).strip()
            codex["authed"] = "logged in" in line.lower()
            codex["auth_detail"] = line.splitlines()[0] if line else ""
        except Exception:  # noqa: BLE001
            codex["authed"] = None
    codex["ready"] = bool(codex_path) and codex.get("authed") is True
    out["codex"] = codex

    agy_path = _which(AGY_BIN)
    agy: dict = {"installed": bool(agy_path), "path": agy_path,
                 "quota": "Google AI Pro subscription (shared Google pool)"}
    if agy_path:
        try:
            v = subprocess.run([agy_path, "--version"], capture_output=True,
                               text=True, timeout=15)
            agy["version"] = (v.stdout or v.stderr).strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
        creds = Path.home() / ".antigravity"
        gcreds = Path.home() / ".gemini" / "oauth_creds.json"
        agy["authed"] = creds.exists() or gcreds.exists() or bool(
            os.environ.get("ANTIGRAVITY_API_KEY"))
    agy["ready"] = bool(agy_path) and agy.get("authed") is True
    if agy_path and not agy["ready"]:
        agy["hint"] = "Run `agy` once and sign in with your Google (Pro) account."
    out["antigravity"] = agy

    return out
