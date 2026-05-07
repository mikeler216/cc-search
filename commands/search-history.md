---
description: "Semantic search over past Claude Code conversations"
argument-hint: "QUERY [--all] [--top N]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh:*)"]
---

Search past conversations using the `cc-search` CLI. Results include resume commands to jump back into any conversation.

IMPORTANT: Always display the FULL session ID (complete UUID) in results and resume commands. Never truncate or shorten session IDs — short hashes are useless for `claude --resume`.

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/run-cc-search.sh" query $ARGUMENTS --top 5
```
