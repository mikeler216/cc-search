#!/usr/bin/env bash
set -e

echo "cc-search installer"
echo "==================="
echo ""
echo "This will install:"
echo "  1. uv (Python package manager) — if not already installed"
echo "  2. cc-search CLI (semantic search over Claude Code conversations)"
echo "  3. Build the search index from your conversation history"
echo ""
read -p "Continue? [Y/n] " answer
if [[ "$answer" =~ ^[Nn] ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""

# Step 1: uv
if command -v uv &>/dev/null; then
  echo "✓ uv already installed"
else
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "✓ uv installed"
fi

# Step 2: cc-search
echo "Installing cc-search..."
uv tool install "git+https://github.com/mikeler216/cc-search"
echo "✓ cc-search installed"

# Step 3: Index
echo ""
echo "Building search index (this may take a minute on first run)..."
cc-search index
echo ""
cc-search status
echo ""
echo "Done! Use /cc-search:search-history in Claude Code to search your conversations."
