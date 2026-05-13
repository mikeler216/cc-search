---
name: search-history
description: Semantic search over past Claude Code conversations. Use when the user wants to find a previous conversation, recall how something was done before, or search their history. Triggers on "do you remember", "find that conversation", "search history", "when did we", or /search-history.
---

## CRITICAL RULES

- You MUST use the `cc-search` CLI tool. This is a semantic search tool, not grep.
- NEVER search conversation files manually with grep, find, cat, or any other tool.
- NEVER read JSONL files directly from ~/.claude/projects/.
- The ONLY correct approach is running `cc-search query` via Bash.
- Use `${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh` so the plugin installs or replaces any legacy Python `cc-search` with the Go binary in `~/.local/bin/cc-search`.

## Step 1: Check if cc-search is installed

Run via Bash: `"${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh" status`

If status shows 0 chunks, run `cc-search index` to build the index.

## Step 2: Search

Run via Bash — this is the ONLY way to search:

```bash
"${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh" query "<user's search terms>" --top 5
```

To search across ALL projects (not just the current directory):

```bash
"${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh" query "<user's search terms>" --top 5 --all
```

Do NOT modify the query. Pass the user's words directly.

## Step 3: Present results

From the cc-search output, present each result as:
- The matching text snippet
- Project name and role (user/assistant)
- The resume command to jump back into that conversation

IMPORTANT: Always show the FULL session ID (complete UUID) in resume commands and tables. Never truncate or abbreviate session IDs — `claude --resume` requires the full UUID to work.
If `cc-search` says `No strong matches found.`, reply that no matching conversation was found. Do not infer relevance from low scores or invent a "most relevant" session.

Keep it concise. Do not add your own commentary about the results.
