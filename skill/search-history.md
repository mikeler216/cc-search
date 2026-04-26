---
name: search-history
description: Semantic search over past Claude Code conversations. Use when the user wants to find a previous conversation, recall how something was done before, or search their history. Triggers on "do you remember", "find that conversation", "search history", "when did we", or /search-history.
---

Run `cc-search query "<user's search terms>" --top 5` via Bash.

If the command is not found, tell the user:
> `cc-search` is not installed. Install it with:
> ```
> uv tool install /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
> ```

If the output says "No results found", suggest the user run `cc-search index` to build or update the index.

Otherwise, present each result as:
- The matching text snippet
- Project name and whether it was a user message or assistant response
- The resume command so they can jump back into that conversation

Keep presentation concise — just the essential info from each result.
