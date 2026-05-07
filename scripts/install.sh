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

REPO="mikeler216/cc-search"
BINARY="cc-search-${OS}-${ARCH}"

fetch_latest_tag() {
  local body
  if command -v curl &>/dev/null; then
    body="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest")"
  elif command -v wget &>/dev/null; then
    body="$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest")"
  else
    echo "Error: curl or wget required"
    exit 1
  fi

  printf '%s\n' "$body" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1
}

LATEST_TAG="$(fetch_latest_tag)"
if [[ -z "$LATEST_TAG" ]]; then
  echo "Error: could not resolve the latest GitHub release tag"
  exit 1
fi

URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}/${BINARY}"

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

echo "Downloading cc-search ${LATEST_TAG} for ${OS}/${ARCH}..."
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
