---
description: "Semantic search over past Claude Code conversations"
argument-hint: "QUERY [--all] [--top N]"
allowed-tools: ["Bash(cc-search:*)", "Bash(uv:*)", "Bash(curl:*)"]
---

Search past conversations using the `cc-search` CLI. Results include resume commands to jump back into any conversation.

IMPORTANT: Always display the FULL session ID (complete UUID) in results and resume commands. Never truncate or shorten session IDs — short hashes are useless for `claude --resume`.

```!
if ! command -v cc-search &>/dev/null; then
  if ! command -v uv &>/dev/null; then
    echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
  fi
  echo "Installing cc-search..." && uv tool install "git+https://github.com/mikeler216/cc-search" && echo "Building search index (first run)..." && cc-search index
fi
cc-search query $ARGUMENTS --top 5
```
