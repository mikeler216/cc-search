# Claude Code Conversation Semantic Search — Design Spec

## Summary

A local-only Python CLI tool (`cc-search`) that provides semantic search over Claude Code conversation history. It indexes full conversations (user + assistant turns) from `~/.claude/projects/**/*.jsonl`, embeds them with `all-MiniLM-L6-v2`, stores vectors in SQLite + `sqlite-vec`, and returns nearest results with `claude --resume` links. Accessible as a standalone CLI and a Claude Code skill (`/search-history`).

## Architecture

```
~/.claude/projects/**/*.jsonl   (conversation source)
         │
         ▼
   ┌─────────────┐
   │  cc-search   │  Python CLI (uv-managed)
   │   index      │  Reads JSONL → chunks into turns → embeds → stores
   │   query      │  Cosine similarity search → ranked results
   └──────┬──────┘
          │
   ┌──────┴──────┐
   │  SQLite DB   │  ~/.claude/search-index/conversations.db
   │  + sqlite-vec│  Stores: embeddings, text chunks, metadata
   └─────────────┘
          │
    Two consumers:
    ├── CLI: `cc-search query "how did I set up auth?"`
    └── Skill: `/search-history how did I set up auth?`
```

### Components

- **Indexer** (`indexer.py`) — Parses JSONL files, splits conversations into chunks (per-turn or multi-turn windows), generates embeddings with `all-MiniLM-L6-v2`, stores in SQLite with `sqlite-vec` for vector search.
- **Searcher** (`searcher.py`) — Takes a query, embeds it, runs nearest-neighbor search via `sqlite-vec`, returns top-k results with text, project, session ID, and `claude --resume <session-id>` command.
- **DB** (`db.py`) — SQLite + `sqlite-vec` setup, schema management, queries.
- **Chunker** (`chunker.py`) — Turn splitting, overlap windowing for long turns.
- **CLI** (`cli.py`) — Click-based CLI with `index`, `query`, and `status` subcommands.
- **Skill** (`skill/search-history.md`) — Claude Code skill that wraps the CLI.

## Data Model

### SQLite Schema

```sql
-- Track which files have been indexed and when
CREATE TABLE files (
  path TEXT PRIMARY KEY,
  last_modified REAL,
  last_indexed REAL
);

-- Conversation chunks with metadata
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  file_path TEXT,          -- source JSONL file
  session_id TEXT,         -- for --resume
  project TEXT,            -- project directory name
  role TEXT,               -- 'user' or 'assistant'
  chunk_text TEXT,         -- the actual text content
  turn_index INTEGER,      -- position in conversation
  created_at REAL          -- timestamp if available
);

-- Vector embeddings (sqlite-vec virtual table)
CREATE VIRTUAL TABLE chunks_vec USING vec0 (
  id INTEGER PRIMARY KEY,
  embedding FLOAT[384]     -- all-MiniLM-L6-v2 output dimension
);
```

### Chunking Strategy

- Each conversation turn (one user message or one assistant response) = one chunk.
- Long turns (>512 tokens) get split into overlapping windows (400 token chunks, 100 token overlap) to stay within the embedding model's sweet spot.
- Metadata (session ID, project, role) extracted from the JSONL structure and the session files in `~/.claude/sessions/`.

### Incremental Indexing

- On `cc-search index`, compare each JSONL file's `last_modified` against `files.last_indexed`.
- Only re-process files that are new or changed.
- Track byte offset for append-only JSONL files to only read new lines.

## Project Structure

```
claude-conversation-semantic-search/
├── pyproject.toml          # uv project, dependencies + entry points
├── src/
│   └── cc_search/
│       ├── __init__.py
│       ├── cli.py          # Click CLI: index, query, status
│       ├── indexer.py      # JSONL parsing, chunking, embedding
│       ├── searcher.py     # Query embedding + sqlite-vec search
│       ├── db.py           # SQLite + sqlite-vec setup & queries
│       └── chunker.py      # Turn splitting, overlap windowing
└── skill/
    └── search-history.md   # Claude Code skill file
```

### Dependencies

- `sentence-transformers` — embedding model (`all-MiniLM-L6-v2`)
- `sqlite-vec` — vector search extension for SQLite
- `click` — CLI framework

### Packaging (uv)

```bash
uv tool install .                    # from local repo
uv tool install git+https://...      # from remote
```

Provides a globally available `cc-search` command.

## CLI Interface

```bash
# Build/update the index (incremental by default)
cc-search index
cc-search index --full          # force full reindex
cc-search index --watch         # keep running, reindex on file changes

# Search conversations
cc-search query "how did I configure auth middleware"
cc-search query "that bug with the database migration" --top 5
cc-search query "react component" --project server-infra

# Status
cc-search status                # index stats: total chunks, last indexed, DB size
```

### Query Output Format

```
── Result 1 (score: 0.87) ──────────────────────────
Project: server-infra
Role:    user
Turn:    12

  "I need to set up auth middleware that validates JWT tokens
   and checks against our user service..."

  → claude --resume a3f2c1d9-4e5b-...

── Result 2 (score: 0.82) ──────────────────────────
Project: server-infra
Role:    assistant
Turn:    13

  "Here's how to set up the JWT validation middleware.
   First, install the dependencies..."

  → claude --resume a3f2c1d9-4e5b-...
```

Default: top 5 results. `--project` filters by project. `--role user|assistant` filters by speaker.

## Skill Definition

```markdown
---
name: search-history
description: Semantic search over past Claude Code conversations.
  Use when the user wants to find a previous conversation, recall
  how something was done before, or search their history.
---

Run `cc-search query "<user's search terms>" --top 5` via Bash.

Present each result as:
- The matching text snippet (trimmed to key content)
- Project name and role (user/assistant)
- The resume command: `claude --resume <session-id>`

If cc-search is not found, tell the user to install it:
  `uv tool install <path-to-project>`

If the index is empty, suggest running `cc-search index` first.
```

Triggered by `/search-history <query>` or naturally when a user asks things like "do you remember when we..." or "find that conversation where...".

## Performance & Constraints

| Aspect | Detail |
|--------|--------|
| Embedding model | `all-MiniLM-L6-v2`, 384 dims, ~80MB |
| Cold start | ~1-2s model load on first query |
| Index time (first run) | <30s for ~2,300 history entries |
| Incremental index | <1s for new/changed files |
| Search latency | <50ms (sqlite-vec brute-force on thousands of chunks) |
| DB location | `~/.claude/search-index/conversations.db` |
| DB size | ~5-10MB for thousands of turns |
| Model cache | `~/.cache/torch/sentence_transformers/` |
| Scale ceiling | ~100k chunks (years of heavy usage) before needing ANN |
| Watch mode | Polls every 30s, no filesystem event dependency |
