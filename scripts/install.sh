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
  curl -L -o /tmp/cc-search.sha256 "${URL}.sha256"
elif command -v wget &>/dev/null; then
  wget -O "${INSTALL_DIR}/cc-search" "$URL"
  wget -O /tmp/cc-search.sha256 "${URL}.sha256"
else
  echo "Error: curl or wget required"
  exit 1
fi

echo "Verifying checksum..."
if command -v shasum &>/dev/null; then
  expected="$(awk '{print $1}' /tmp/cc-search.sha256)  ${INSTALL_DIR}/cc-search"
  echo "$expected" | shasum -a 256 -c - || { echo "Error: checksum verification failed"; rm -f "${INSTALL_DIR}/cc-search"; exit 1; }
elif command -v sha256sum &>/dev/null; then
  expected="$(awk '{print $1}' /tmp/cc-search.sha256)  ${INSTALL_DIR}/cc-search"
  echo "$expected" | sha256sum -c - || { echo "Error: checksum verification failed"; rm -f "${INSTALL_DIR}/cc-search"; exit 1; }
else
  echo "Warning: neither shasum nor sha256sum found; skipping checksum verification"
fi
rm -f /tmp/cc-search.sha256

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
