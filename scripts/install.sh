#!/usr/bin/env bash
set -e

echo "cc-search installer"
echo "==================="
echo ""
echo "This will install:"
echo "  1. cc-search binary (semantic search over Claude Code conversations)"
echo "  2. Build the search index from your conversation history"
echo ""
read -p "Continue? [Y/n] " answer
if [[ "$answer" =~ ^[Nn] ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac

BINARY="cc-search-${OS}-${ARCH}"
URL="https://github.com/mikeler216/cc-search/releases/latest/download/${BINARY}"

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

echo "Downloading cc-search for ${OS}/${ARCH}..."
if command -v curl &>/dev/null; then
  curl -L -o "${INSTALL_DIR}/cc-search" "$URL"
elif command -v wget &>/dev/null; then
  wget -O "${INSTALL_DIR}/cc-search" "$URL"
else
  echo "Error: curl or wget required"
  exit 1
fi

chmod +x "${INSTALL_DIR}/cc-search"
echo "✓ cc-search installed to ${INSTALL_DIR}/cc-search"

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  echo ""
  echo "Add to your PATH: export PATH=\"$INSTALL_DIR:\$PATH\""
fi

echo ""
echo "Building search index (this may take a minute on first run)..."
"${INSTALL_DIR}/cc-search" index
echo ""
"${INSTALL_DIR}/cc-search" status
echo ""
echo "Done! Use /cc-search:search-history in Claude Code to search your conversations."
