---
description: "Semantic search over past Claude Code conversations"
argument-hint: "QUERY [--all] [--top N]"
allowed-tools: ["Bash(cc-search:*)"]
---

Search past conversations using the `cc-search` CLI. Results include resume commands to jump back into any conversation.

```!
cc-search query $ARGUMENTS --top 5
```
