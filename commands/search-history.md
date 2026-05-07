---
description: "Semantic search over past Claude Code conversations"
argument-hint: "QUERY [--all] [--top N]"
allowed-tools: ["Bash(cc-search:*)", "Bash(curl:*)", "Bash(wget:*)"]
---

Search past conversations using the `cc-search` CLI. Results include resume commands to jump back into any conversation.

IMPORTANT: Always display the FULL session ID (complete UUID) in results and resume commands. Never truncate or shorten session IDs — short hashes are useless for `claude --resume`.

```!
if ! command -v cc-search &>/dev/null; then
  OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH="$(uname -m)"
  case "$ARCH" in x86_64) ARCH="amd64" ;; aarch64|arm64) ARCH="arm64" ;; esac
  BINARY="cc-search-${OS}-${ARCH}"
  URL="https://github.com/mikeler216/cc-search/releases/latest/download/${BINARY}"
  mkdir -p "$HOME/.local/bin"
  echo "Installing cc-search..." && curl -L -o "$HOME/.local/bin/cc-search" "$URL" && chmod +x "$HOME/.local/bin/cc-search"
  export PATH="$HOME/.local/bin:$PATH"
  echo "Building search index (first run)..." && cc-search index
fi
cc-search query $ARGUMENTS --top 5
```
