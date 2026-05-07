# cc-search Go Rewrite — Design Spec

## Motivation

Rewrite cc-search from Python to Go for two reasons:

1. **Distribution** — eliminate the Python/uv runtime dependency. Ship a single static binary that users download and run.
2. **Performance** — faster cold start, lower memory footprint, faster indexing.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding inference | ONNX Runtime via `onnxruntime-go` (CGo) | Same model quality, fully local, mature binding |
| Vector search | SQLite + `sqlite-vec` via `mattn/go-sqlite3` (CGo) | Proven in current version, SIMD-optimized cosine similarity |
| Model bundling | `//go:embed` into binary | Single file, zero network needed, true "download and go" |
| Tokenizer | Pure Go WordPiece | ~150 lines, no extra C dependency, vocab embedded alongside model |
| Chunking | Token-aware (real WordPiece token count) | More accurate boundaries than whitespace splitting |
| CLI framework | Cobra | Standard Go CLI library, maps cleanly to existing interface |
| DB compatibility | Fresh schema, reindex on migration | ONNX tokenizer may differ from Python's; DB is a derived cache anyway |
| Distribution | GitHub Releases with pre-built binaries per platform | macOS arm64/amd64, Linux amd64 |

## Project Structure

```
cc-search/
├── cmd/cc-search/
│   └── main.go              # Cobra root + subcommands
├── internal/
│   ├── embedding/           # ONNX inference + WordPiece tokenizer
│   │   ├── embedding.go     # Load model, Embed([]string) → [][]float32
│   │   └── tokenizer.go     # WordPiece: Tokenize, CountTokens
│   ├── store/               # SQLite + sqlite-vec persistence
│   │   └── store.go         # Open, UpsertFile, InsertChunks, Search, Stats
│   ├── ingest/              # File discovery, JSONL parsing, chunking, orchestration
│   │   └── ingest.go        # Run(): find → parse → chunk → embed → store
│   └── search/              # Query embedding + vector search + result formatting
│       └── search.go        # Run(): query string → []Result
├── assets/                  # Embedded at compile time
│   ├── model.onnx           # all-MiniLM-L6-v2 ONNX export (~30MB)
│   └── vocab.txt            # WordPiece vocabulary
├── scripts/install.sh       # Detect platform, download binary from GitHub Releases
├── skills/                  # Claude Code plugin skills (unchanged)
├── commands/                # Claude Code commands (unchanged)
├── .claude-plugin/          # Plugin config (bump version)
└── .github/workflows/
    ├── ci.yml               # Build + test per platform
    └── release.yml          # Cross-compile + upload to GitHub Releases
```

### Package responsibilities

**`internal/embedding`** — Owns the ONNX model and the WordPiece tokenizer. Single public type `Model` with methods `Embed([]string) ([][]float32, error)` and `CountTokens(string) int`. At init, writes the embedded ONNX bytes to a temp file (ONNX Runtime needs a file path), loads the session once, reuses for the process lifetime. Batch inference groups chunks (e.g., 64 at a time) to amortize session overhead.

**`internal/store`** — The only package that touches SQLite. Manages schema creation, version checks, file tracking, chunk storage, vector search, and stats. Loads `sqlite-vec` as a compiled-in extension. Exposes `Open(path) (*DB, error)` and methods on `*DB`.

**`internal/ingest`** — Orchestrates the indexing pipeline. Discovers `~/.claude/projects/**/*.jsonl` files, filters by mtime for incremental indexing, parses JSONL lines, extracts messages (handles string and list content formats), chunks text using token-aware splitting (400 tokens max, 100 overlap), embeds chunks via `embedding.Model`, and writes to `store.DB`. Single entry point: `Run()`.

**`internal/search`** — Orchestrates the query path. Embeds the query string, calls `store.Search()` for vector similarity, attaches `claude --resume <session-id>` commands to results. Single entry point: `Run()`.

### Dependency graph

```
cmd/cc-search
  ├── internal/ingest
  │     ├── internal/embedding
  │     └── internal/store
  └── internal/search
        ├── internal/embedding
        └── internal/store
```

No circular dependencies. `ingest` and `search` never depend on each other.

## Database Schema

```sql
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE files (
    path          TEXT PRIMARY KEY,
    last_modified REAL NOT NULL,
    last_indexed  REAL NOT NULL
);

CREATE TABLE chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path  TEXT NOT NULL REFERENCES files(path),
    session_id TEXT NOT NULL,
    project    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    chunk_text TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    created_at REAL
);

CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_project ON chunks(project);

CREATE VIRTUAL TABLE chunks_vec USING vec0 (
    id        INTEGER PRIMARY KEY,
    embedding FLOAT[384]
) WITH (distance_metric=cosine);
```

The `meta` table stores `model_hash` (SHA256 of the embedded ONNX model bytes) and `schema_version`. These are checked on every DB open to detect when a rebuild is needed.

## Auto-rebuild on Version Mismatch

On every command that opens the DB:

1. Read `meta.schema_version` and `meta.model_hash`
2. Compare against the compiled-in values
3. If both match: proceed normally
4. If either differs on `index` or `update`: drop all tables, recreate schema, run a full reindex automatically
5. If either differs on `query` or `status`: print a warning ("index out of date, run `cc-search index` to rebuild") and exit — don't silently reindex during a read-only command

This eliminates the need for migration scripts. The DB is a derived cache — the JSONL files are the source of truth.

## CLI Interface

Same commands and flags as the Python version:

```
cc-search index [--db-path PATH] [--claude-dir DIR] [--full] [--watch]
cc-search query <terms...> [--db-path PATH] [--top N] [--all] [--project PATH] [--role user|assistant]
cc-search status [--db-path PATH]
cc-search update
```

### Behavior changes from Python version

- **`index --watch`** uses `fsnotify` for file system events instead of 30s polling
- **`update`** downloads a platform-specific binary from GitHub Releases, self-replaces via atomic rename (write to temp file next to current binary, then `os.Rename` over it), then execs the new binary with `index --full` to reindex (since the embedded model may have changed)

### Defaults

| Setting | Value |
|---------|-------|
| DB path | `~/.claude/search-index/conversations.db` |
| Claude dir | `~/.claude` |
| Top results | 5 |
| Chunk max tokens | 400 |
| Chunk overlap | 100 |
| Embedding dimensions | 384 |

## Embedding Details

**ONNX model:** `all-MiniLM-L6-v2` exported to ONNX format. Inputs: `input_ids`, `attention_mask`, `token_type_ids` (int64 tensors). Output: pooled 384-dimensional float32 vector.

**WordPiece tokenizer (pure Go):** Loads `vocab.txt` into a map at init. Algorithm: lowercase → split on whitespace → split on punctuation → greedy longest-match subword lookup → `[UNK]` fallback. Adds `[CLS]` prefix and `[SEP]` suffix for model input.

**Migration from Python:** Reindex once. The ONNX model weights are identical to PyTorch, but the Go WordPiece tokenizer may produce subtly different tokenization than HuggingFace's Python tokenizer. After migration, embeddings are deterministic within the Go binary.

**Go-to-Go updates:** No reindex needed unless `model_hash` changes. Tracked automatically via the `meta` table.

## Build & Release

**Binary composition:**

| Component | Size |
|-----------|------|
| Go binary | ~10MB |
| ONNX model (embedded) | ~30MB |
| ONNX Runtime (linked) | ~8MB |
| sqlite3 + sqlite-vec (linked) | ~2MB |
| **Total** | **~50MB** |

**CI (`ci.yml`):**
- Matrix: `macos-arm64`, `macos-amd64`, `linux-amd64`
- Steps: install ONNX Runtime SDK, `go test ./...`, `go vet`, `golangci-lint`
- Tests use temp DB and embedded model — no external downloads

**Release (`release.yml`):**
- Triggered on git tag (`v*`)
- Native build per platform (no CGo cross-compilation)
- Upload binaries: `cc-search-darwin-arm64`, `cc-search-darwin-amd64`, `cc-search-linux-amd64`
- Generate SHA256 checksums

**install.sh:**
- Detect OS and architecture
- Download correct binary from latest GitHub Release
- Place in `~/.local/bin` or `/usr/local/bin`
- Run `cc-search index` to build initial index

## Plugin Integration

Plugin files remain unchanged in structure:

- `skills/search-history/SKILL.md` — calls `cc-search query`
- `skills/update-cli/SKILL.md` — calls `cc-search update`
- `commands/search-history.md` — command wrapper
- `.claude-plugin/plugin.json` — bump version, update install instructions to point at binary download
- `scripts/install.sh` — rewritten to download binary instead of `uv tool install`
