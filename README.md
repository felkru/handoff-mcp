# handoff-mcp

An MCP server (and matching CLI) that lets **Claude hand a task off to a peer
coding-agent CLI** — OpenAI's **Codex** or Google's **Antigravity** (Gemini 3) —
and get the answer back inline.

Why you'd want this:

- **Independent second / adversarial opinion.** Get a genuinely different model
  to poke holes in a proof, design, diff, or argument instead of Claude
  reviewing its own work.
- **Separate quota pools.** Codex runs on your OpenAI/ChatGPT subscription;
  Antigravity runs on your Google AI Pro subscription. Handing work off
  **preserves your Claude usage limit at no extra cost** — the two peers don't
  touch each other's quota or Claude's.
- **Visual feedback.** Antigravity/Gemini 3 is natively multimodal — attach a
  screenshot, diagram, chart, or PDF and ask what's wrong with it.
- **Async status.** Fire a long handoff as a background job and poll it; you see
  the peer agent's progress stream in rather than blocking on one long call.

> **Be aggressive about handing off when your Claude usage limit is near.** The
> server's tool docs instruct the calling model to lower its bar for delegating
> when the Claude/Fable limit is close — bulk review, long reads, second
> opinions, and mechanical agentic coding are all good candidates.

---

## Backends — who's good at what

The routing guidance below is **verified as of 2026-07-21**. The model landscape
moves fast; see [Maintenance](#maintenance).

| | **codex** (`codex exec`) | **antigravity** (`agy`) |
|---|---|---|
| Model | OpenAI GPT-5.x-Codex | Google Gemini 3 Pro |
| Quota | OpenAI / ChatGPT subscription | Google AI Pro subscription |
| **Best at** | agentic/terminal coding (Terminal-Bench ~77% vs ~54%), long-horizon autonomous work, large refactors & migrations, adversarial **code** review, bug hunting, long-context reliability | **visual** feedback (screenshots, diagrams, PDFs), **very long** context (1M–2M tokens: whole monorepos / long docs), novel reasoning, cheap bulk work |
| **Weak at** | images / diagrams / UI (text-specialized); Codex surface caps context ~400K | terminal-agentic coding; the `agy` CLI is finicky headless (worked around here) |
| Attachments | text + images | text + images + PDFs |

**Rule of thumb:** code / agentic / second-opinion-on-text → **codex**. Anything
visual, or too large for a normal context window → **antigravity**.

> Note: Google retired the open-source `gemini` CLI for individual accounts on
> **2026-06-18** (`IneligibleTierError`); Pro/Ultra/free users are served by
> Antigravity now. That's why the Google backend here is `agy`, not `gemini`.

---

## Tools

| Tool | Purpose |
|------|---------|
| `handoff` | Run a backend **synchronously**, return its answer. |
| `handoff_start` | Fire a **background job**, return a job id immediately. |
| `handoff_status` | Poll a job: `state`, `elapsed_s`, streamed `log_tail`, `result`. |
| `handoff_jobs` | List recent jobs. |
| `handoff_backends` | Install/auth readiness of each backend + a guidance-staleness banner. |
| `handoff_guidance` | The full model routing guide. |

### `handoff(prompt, backend, files=[], model="", cwd=None, approve=False, timeout=600)`

- `prompt` — self-contained instruction. The peer has **none** of your
  conversation's context.
- `backend` — `"codex"` or `"antigravity"`.
- `files` — absolute/`~` paths. codex: text + images. antigravity: text, images,
  PDFs. Attachments are **copied** into a throwaway temp workspace — originals
  are never touched.
- `model` — optional id override; empty = the CLI's own default.
- `cwd` — a project dir the peer may read for extra context (never written
  unless `approve=True`).
- `approve` — allow the peer to make edits (codex `workspace-write`; antigravity
  `accept-edits`). Default `False` = read-only, right for reviews/opinions.
- `timeout` — seconds before abort.

### Async pattern

```
job = handoff_start("deep-review this repo for races", "codex", cwd="~/proj")
# → "Started codex job a1b2c3…"
handoff_status("a1b2c3…")   # poll until state != "running"
```

---

## Setup

### 1. Install the peer CLIs

**Codex** (OpenAI):
```bash
brew install codex        # or: npm i -g @openai/codex
codex login               # sign in with your ChatGPT account
codex login status        # → "Logged in using ChatGPT"
```

**Antigravity** (Google):
```bash
brew install --cask antigravity-cli
agy                       # run once, sign in with your Google (Pro) account, then quit
```

### 2. Install this server

```bash
uv tool install --force ~/Documents/programming/handoff-mcp
```

Installs both the `handoff` CLI and the `handoff-mcp` server onto your PATH.

### 3. Register the MCP server with Claude Code

```bash
claude mcp add handoff -- handoff-mcp
```

Check everything's wired up:

```bash
handoff --status      # backend install/auth + guidance freshness
handoff --guidance    # the routing guide
```

---

## CLI usage

```bash
# adversarial code review
handoff codex "review this for correctness and edge cases" -f src/core.py

# visual feedback
handoff antigravity "what's wrong with this dashboard layout?" -f screenshot.png

# read something huge
handoff antigravity "summarize the key obligations in this contract" -f contract.pdf

# let a peer explore a whole repo (read-only)
handoff codex "is there a race condition in the job queue?" -C ~/Documents/programming/myproj

# actually let it edit (opt-in)
handoff codex "migrate this file to async/await" -f app.py --approve

# long job, tail progress until done
handoff codex "find every N+1 query" -C ~/proj --async

# health / guidance
handoff --status
handoff --guidance
```

---

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `HANDOFF_CODEX_BIN` | `codex` | Path/name of the Codex binary. |
| `HANDOFF_AGY_BIN` | `agy` | Path/name of the Antigravity binary. |
| `HANDOFF_MODEL` | *(empty)* | Default model override for both backends. |
| `HANDOFF_TIMEOUT` | `600` | Default sync timeout (seconds). |
| `HANDOFF_JOB_TIMEOUT` | `1800` | Default background-job timeout (seconds). |
| `HANDOFF_MCP_HOME` | `~/.cache/handoff-mcp` | Where job logs live. |

---

## How it works

All logic lives in `src/handoff_mcp/core.py`; the MCP server (`server.py`) and
CLI (`cli.py`) are thin, DRY wrappers.

- **Codex** is headless-native: we run `codex exec --json --skip-git-repo-check
  -s <sandbox> -o <last-message-file>`, stream the JSONL events into a per-job
  log for status, and read the clean final answer from the `-o` file. The prompt
  is fed via **stdin** (codex's `-i/--image` flag is variadic and would swallow a
  trailing positional prompt).
- **Antigravity** (`agy`) gates its stdout on `isatty()` (upstream bug: empty
  output in a non-TTY). We run it **inside a pseudo-terminal** and strip the ANSI
  / spinner noise back out. It also can't prompt for tool permissions headlessly,
  so we pass `--dangerously-skip-permissions` (safe: throwaway temp workspace
  with only file *copies*) and gate edits with `--mode plan` (read-only) vs
  `accept-edits`.
- **Async jobs** run on background threads writing to a log file, so
  `handoff_status` can return live progress and the final result across
  successive tool calls.

## Maintenance

The strengths/weaknesses and default models reflect the model landscape as of
`core.MODEL_INFO_UPDATED` (**2026-07-21**). `handoff_backends` and
`handoff --status` print a **staleness banner** that escalates to a warning once
the guidance is over ~120 days old. When new Gemini / GPT / Codex versions ship,
update `ROUTING_GUIDE` and `MODEL_INFO_UPDATED` in `src/handoff_mcp/core.py`.

## License

MIT
