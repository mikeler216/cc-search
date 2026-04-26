---
name: search-history
description: Semantic search over past Claude Code conversations. Use when the user wants to find a previous conversation, recall how something was done before, or search their history. Triggers on "do you remember", "find that conversation", "search history", "when did we", or /search-history.
---

## Prerequisites Check

First, check if `cc-search` is installed by running `cc-search --help` via Bash.

If NOT installed, tell the user and run the setup:

```bash
uv tool install "git+https://github.com/michaeller/claude-conversation-semantic-search"
cc-search index
```

If installed but the index is empty (check with `cc-search status`), run `cc-search index` first.

## Search

Run via Bash:

```bash
cc-search query "<user's search terms>" --top 5
```

To search across ALL projects (not just the current directory):

```bash
cc-search query "<user's search terms>" --top 5 --all
```

## Presenting Results

Present each result as:
- The matching text snippet
- Project name and whether it was a user message or assistant response
- The resume command so they can jump back into that conversation

Keep presentation concise.
