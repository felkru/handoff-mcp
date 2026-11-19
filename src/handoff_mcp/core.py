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
# Kill a backend that produces no output at all for this long (a stalled CLI,
# e.g. codex losing its API connection, used to burn the whole JOB_TIMEOUT).
IDLE_TIMEOUT = int(os.environ.get("HANDOFF_IDLE_TIMEOUT", "480"))

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


class HandoffExhausted(HandoffError):
    """Every eligible backend refused for quota/auth reasons, not task reasons.

    Distinct from HandoffError because the caller's correct response differs: a
    task failure should be read and reacted to, whereas exhaustion means no peer
    agent is available at all and the work must go to a native Claude subagent.
    """


# Returned verbatim to the model when no backend can be reached. The tool
# descriptions bind the model to switch to the native Agent tool on sight of it;
# an MCP server cannot invoke Agent itself, so this string IS the handover.
FALLBACK_MARKER = "[handoff exhausted → use native Agent]"

# Substrings that mean "this backend won't work for anyone right now" as opposed
# to "this task failed". Kept deliberately broad: a missed match wastes a
# rotation attempt, while a false positive would hide a genuine task error —
# so patterns must not include generic words like "error" or "failed" alone.
_EXHAUSTION_RE = re.compile(
    r"usage limit|quota|rate.?limit|too many requests|\b429\b|out of credits"
    r"|insufficient (?:credits|balance)|plan limit|resource[_ ]exhausted"
    r"|not logged in|unauthorized|\b401\b|authentication (?:failed|required)"
    r"|please (?:re-?)?log ?in|log in again|session expired",
    re.I,
)


def _is_exhaustion(msg: str) -> bool:
    return bool(_EXHAUSTION_RE.search(msg or ""))


def _eligible(backend: str, files: list[str] | None) -> bool:
    """Can `backend` even accept these attachments?

    Only antigravity reads PDFs, so rotating a PDF task onto codex would trade a
    quota error for a guaranteed capability failure. Rotation must skip it.
    """
    for raw in files or []:
        if Path(raw).suffix.lower() == ".pdf" and backend == "codex":
            return False
    return True


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
        effort = os.environ.get("HANDOFF_CODEX_EFFORT", "").strip()
        if effort:
            argv += ["-c", f"model_reasoning_effort={effort}"]
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
            # Deliberately avoids the words the exhaustion classifier keys on
            # ("quota", "usage limit", …): this branch fires for *any* empty
            # response, so naming them here would make every agy hiccup look
            # like an out-of-quota event and trigger a pointless rotation.
            raise HandoffError(
                "agy returned no output. Check `handoff_backends` for login and "
                "subscription state, and see the log tail for what it printed."
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
    import select

    start = time.time()
    last_output = start
    fh = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        assert proc.stdout is not None
        while True:
            r, _, _ = select.select([proc.stdout], [], [], 1.0)
            if r:
                line = proc.stdout.readline()
                if not line:
                    break  # EOF
                last_output = time.time()
                if fh:
                    fh.write(_codex_event_line(line))
                    fh.flush()
            elif proc.poll() is not None:
                break
            now = time.time()
            if now - start > timeout:
                proc.kill()
                raise HandoffError(f"codex timed out after {timeout}s.")
            if now - last_output > IDLE_TIMEOUT:
                proc.kill()
                raise HandoffError(
                    f"codex stalled: no output for {IDLE_TIMEOUT}s (killed after "
                    f"{int(now - start)}s). Check network / `codex login status`.")
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
    last_output = start
    fh = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        while True:
            now = time.time()
            if now - start > timeout:
                proc.kill()
                raise HandoffError(f"agy timed out after {timeout}s.")
            if now - last_output > IDLE_TIMEOUT:
                proc.kill()
                raise HandoffError(
                    f"agy stalled: no output for {IDLE_TIMEOUT}s (killed after "
                    f"{int(now - start)}s). Check `handoff_backends` / login state.")
            r, _, _ = select.select([master], [], [], 0.5)
            if master in r:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if not data:
                    break
                last_output = time.time()
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
                timeout=DEFAULT_TIMEOUT, approve=False, log_path=None,
                rotate=True) -> str:
    """Run `backend`, falling back to the other one if this one is out of quota.

    The two backends bill to separate subscription pools, so an exhausted quota
    on one says nothing about the other — rotating is nearly free and usually
    rescues the call. Rotation happens ONLY on exhaustion-shaped failures: a
    genuine task error is re-raised immediately, because silently re-running a
    failing task on a second model would double the cost and muddy the report.

    Raises HandoffExhausted (message begins with FALLBACK_MARKER) when no
    eligible backend is reachable, which is the caller's signal to fall back to
    a native Claude subagent.
    """
    if backend not in _RUNNERS:
        raise HandoffError(f"Unknown backend {backend!r}. Choose one of {BACKENDS}.")
    if not prompt or not prompt.strip():
        raise HandoffError("prompt must be non-empty")

    order = [backend]
    if rotate:
        order += [b for b in BACKENDS if b != backend and _eligible(b, files)]

    refusals: list[str] = []
    for attempt, b in enumerate(order):
        try:
            return _RUNNERS[b](
                prompt, files=files or [], model=model if b == backend else "",
                cwd=cwd, timeout=timeout, log_path=log_path, approve=approve,
            )
        except HandoffError as e:
            if not _is_exhaustion(str(e)):
                raise  # real task failure — surface it, never mask by rotating
            refusals.append(f"{b}: {e}")
            if attempt + 1 < len(order):
                log_note = f"\n[handoff] {b} out of quota, rotating to {order[attempt + 1]}\n"
                if log_path:
                    try:
                        with open(log_path, "a", encoding="utf-8") as fh:
                            fh.write(log_note)
                    except OSError:
                        pass

    raise HandoffExhausted(
        FALLBACK_MARKER
        + "\nEvery peer backend refused for quota/auth reasons:\n  "
        + "\n  ".join(refusals)
        + "\nRe-run this task with the native Agent tool instead; do not retry handoff."
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
    group: str | None = None   # set when the job is part of a start_group fan-out
    label: str = ""            # human tag so a fan-out result set stays readable

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
# Signalled whenever any job reaches a terminal state, so wait_job() can block
# instead of the caller hot-polling get_job().
_DONE = threading.Condition(_LOCK)


def _state_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _sentinel_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.done"


def _persist(job: Job) -> None:
    """Mirror a job's state to disk.

    Two artefacts, both load-bearing:
      * ``<id>.json``  — full state, so status survives an MCP restart.
      * ``<id>.done``  — empty sentinel written only at a terminal state, so an
        external waiter can block on file existence alone (``until [ -f … ]``).
        Agent harnesses notify on a *finishing* shell command, not on a polling
        loop that never exits — without this sentinel there is nothing cheap to
        wait on and a background waiter spins forever.
    """
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        d = asdict(job)
        d["elapsed_s"] = round((job.finished or time.time()) - job.started, 1)
        tmp = _state_path(job.id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
        tmp.replace(_state_path(job.id))
        if job.state != "running":
            _sentinel_path(job.id).write_text(job.state, encoding="utf-8")
    except OSError:
        pass  # disk mirroring is best-effort; the in-memory registry still works


def _worker(job: Job, kwargs: dict) -> None:
    try:
        result = run_backend(job.backend, log_path=Path(job.log_path), **kwargs)
        with _LOCK:
            job.result, job.state, job.finished = result, "done", time.time()
    except Exception as e:  # noqa: BLE001 - surface any failure to the poller
        with _LOCK:
            job.error, job.state, job.finished = str(e), "error", time.time()
    finally:
        _persist(job)
        with _DONE:
            _DONE.notify_all()


def start_job(backend, prompt, *, files=None, model=DEFAULT_MODEL, cwd=None,
              timeout=JOB_TIMEOUT, approve=False, group=None, label="") -> str:
    if backend not in _RUNNERS:
        raise HandoffError(f"Unknown backend {backend!r}. Choose one of {BACKENDS}.")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    log_path = JOBS_DIR / f"{job_id}.log"
    log_path.write_text("", encoding="utf-8")
    job = Job(id=job_id, backend=backend, prompt=prompt[:200],
              started=time.time(), log_path=str(log_path),
              group=group, label=label)
    with _LOCK:
        _JOBS[job_id] = job
    _sentinel_path(job_id).unlink(missing_ok=True)
    _persist(job)
    kwargs = dict(prompt=prompt, files=files, model=model, cwd=cwd,
                  timeout=timeout, approve=approve)
    threading.Thread(target=_worker, args=(job, kwargs), daemon=True).start()
    return job_id


def sentinel_for(job_id: str) -> str:
    """Path of the file that appears the moment `job_id` reaches a terminal state."""
    return str(_sentinel_path(job_id))


def get_job(job_id: str, tail: int = 40) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job:
        d = job.public(tail=tail)
    else:
        # Not in memory: the server restarted since the job was started. The
        # on-disk mirror still has everything except a live worker thread.
        try:
            d = json.loads(_state_path(job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise HandoffError(f"No such job {job_id!r}.") from None
        d["log_tail"] = _tail_file(d.get("log_path", ""), tail)
        if d.get("state") == "running":
            # A restart orphaned the worker; the job can never complete now.
            d["state"] = "orphaned"
            d["error"] = "MCP server restarted while this job was running."
    d["done_sentinel"] = sentinel_for(job_id)
    return d


def wait_job(job_id: str, timeout: float = 1800.0, tail: int = 40) -> dict:
    """Block until `job_id` leaves the running state, then return its status.

    This is the notification path: the tool call itself returns when the job
    finishes, so the caller does not have to guess a polling cadence.
    """
    deadline = time.time() + timeout
    with _DONE:
        while True:
            job = _JOBS.get(job_id)
            if job is None:
                break  # fall through to get_job (disk mirror / clear error)
            if job.state != "running":
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                d = job.public(tail=tail)
                d["done_sentinel"] = sentinel_for(job_id)
                d["wait_timed_out"] = True
                return d
            _DONE.wait(min(remaining, 5.0))
    return get_job(job_id, tail=tail)


def list_jobs(limit: int = 10) -> list[dict]:
    """Recent jobs, merging the in-memory registry with the on-disk mirror.

    Reading only `_JOBS` would make every job vanish from the listing after an
    MCP restart even though `<id>.json` still holds its result — the same
    restart-blindness `get_job` already avoids.
    """
    rows: dict[str, dict] = {}
    try:
        for p in JOBS_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if d.get("state") == "running" and not _sentinel_path(d["id"]).exists():
                d["state"] = "orphaned"  # no live worker can own it after a restart
            rows[d["id"]] = d
    except OSError:
        pass

    with _LOCK:
        for j in _JOBS.values():  # memory wins: it has the live state
            rows[j.id] = asdict(j)

    ordered = sorted(rows.values(), key=lambda d: d.get("started", 0.0), reverse=True)
    return [
        {"id": d["id"], "backend": d.get("backend"), "state": d.get("state"),
         "group": d.get("group"),
         "elapsed_s": round((d.get("finished") or time.time()) - d.get("started", 0.0), 1),
         "prompt": (d.get("prompt") or "")[:80]}
        for d in ordered[:limit]
    ]


# ── fan-out: run many handoffs at once, like several native subagents ─────────

def _group_sentinel(group_id: str) -> Path:
    return JOBS_DIR / f"group-{group_id}.done"


def group_sentinel_for(group_id: str) -> str:
    """Path that appears once EVERY job in the group has reached a terminal state."""
    return str(_group_sentinel(group_id))


def start_group(specs: list[dict], *, backend: str | None = None, cwd=None,
                approve: bool = False, timeout: int = JOB_TIMEOUT) -> dict:
    """Fan a list of tasks out concurrently and return one group handle.

    This is the parity feature for spawning several native subagents at once:
    each spec becomes its own independent job, they all run in parallel, and a
    single group sentinel lets one shell waiter block on the whole batch instead
    of one waiter per job.

    Each spec is a dict with `prompt` (required) and optionally `backend`,
    `files`, `model`, `label`.
    """
    if not specs:
        raise HandoffError("specs must be a non-empty list")
    group_id = uuid.uuid4().hex[:8]
    _group_sentinel(group_id).unlink(missing_ok=True)

    started: list[dict] = []
    for i, spec in enumerate(specs):
        prompt = (spec.get("prompt") or "").strip()
        if not prompt:
            raise HandoffError(f"specs[{i}] has an empty prompt")
        b = spec.get("backend") or backend
        if b not in _RUNNERS:
            raise HandoffError(
                f"specs[{i}]: unknown backend {b!r}. Choose one of {BACKENDS}."
            )
        job_id = start_job(
            b, prompt, files=spec.get("files"), model=spec.get("model", DEFAULT_MODEL),
            cwd=spec.get("cwd", cwd), approve=approve, timeout=timeout,
            group=group_id, label=spec.get("label") or f"task{i + 1}",
        )
        started.append({"job_id": job_id, "backend": b,
                        "label": spec.get("label") or f"task{i + 1}"})

    def _watch() -> None:
        ids = [s["job_id"] for s in started]
        while True:
            with _DONE:
                pending = [i for i in ids
                           if (_JOBS.get(i) and _JOBS[i].state == "running")]
                if not pending:
                    break
                _DONE.wait(5.0)
        try:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            _group_sentinel(group_id).write_text("done", encoding="utf-8")
        except OSError:
            pass

    threading.Thread(target=_watch, daemon=True).start()
    return {"group_id": group_id, "jobs": started,
            "group_sentinel": group_sentinel_for(group_id)}


def get_group(group_id: str, tail: int = 10) -> dict:
    """Status of every job in a group, plus an aggregate state."""
    members = [d for d in (list_jobs(limit=500)) if d.get("group") == group_id]
    if not members:
        raise HandoffError(f"No such group {group_id!r}.")
    detail = []
    for m in members:
        try:
            detail.append(get_job(m["id"], tail=tail))
        except HandoffError:
            detail.append(m)
    running = [d for d in detail if d.get("state") == "running"]
    return {
        "group_id": group_id,
        "aggregate": "running" if running else "finished",
        "n_total": len(detail),
        "n_running": len(running),
        "group_sentinel": group_sentinel_for(group_id),
        "jobs": detail,
    }


def wait_group(group_id: str, timeout: float = 1800.0, tail: int = 10) -> dict:
    """Block until every job in the group is terminal, then return all results."""
    deadline = time.time() + timeout
    while True:
        g = get_group(group_id, tail=tail)
        if g["aggregate"] == "finished":
            return g
        if time.time() >= deadline:
            g["wait_timed_out"] = True
            return g
        with _DONE:
            _DONE.wait(min(5.0, max(0.1, deadline - time.time())))


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
