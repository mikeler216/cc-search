#!/usr/bin/env bash
set -euo pipefail

REPO="${CC_SEARCH_REPO:-mikeler216/cc-search}"
INSTALL_DIR="${CC_SEARCH_INSTALL_DIR:-$HOME/.local/bin}"
TARGET="${INSTALL_DIR}/cc-search"
TMP_SHA256="${CC_SEARCH_TMP_SHA256:-/tmp/cc-search.sha256}"
INSTALLED_THIS_RUN=0

log() {
  echo "$*" >&2
}

detect_os() {
  uname -s | tr '[:upper:]' '[:lower:]'
}

detect_arch() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) echo "$arch" ;;
  esac
}

fetch_latest_tag() {
  local url body
  url="https://api.github.com/repos/${REPO}/releases/latest"
  if command -v curl >/dev/null 2>&1; then
    body="$(curl -fsSL "$url")"
  elif command -v wget >/dev/null 2>&1; then
    body="$(wget -qO- "$url")"
  else
    log "Error: curl or wget required"
    exit 1
  fi

  printf '%s\n' "$body" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1
}

download_binary() {
  local latest_tag os arch binary url tmp_target expected

  mkdir -p "$INSTALL_DIR"

  if [[ -n "${CC_SEARCH_TEST_BINARY_SOURCE:-}" ]]; then
    cp "${CC_SEARCH_TEST_BINARY_SOURCE}" "$TARGET"
    chmod +x "$TARGET"
    INSTALLED_THIS_RUN=1
    return
  fi

  latest_tag="$(fetch_latest_tag)"
  if [[ -z "$latest_tag" ]]; then
    log "Error: could not resolve the latest GitHub release tag"
    exit 1
  fi

  os="$(detect_os)"
  arch="$(detect_arch)"
  binary="cc-search-${os}-${arch}"
  url="https://github.com/${REPO}/releases/download/${latest_tag}/${binary}"
  tmp_target="${TARGET}.download"

  log "Installing Go cc-search ${latest_tag} to ${TARGET}..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$tmp_target" "$url"
    curl -fsSL -o "$TMP_SHA256" "${url}.sha256"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$tmp_target" "$url"
    wget -qO "$TMP_SHA256" "${url}.sha256"
  else
    log "Error: curl or wget required"
    exit 1
  fi

  if command -v shasum >/dev/null 2>&1; then
    expected="$(awk '{print $1}' "$TMP_SHA256")  ${tmp_target}"
    echo "$expected" | shasum -a 256 -c - >/dev/null
  elif command -v sha256sum >/dev/null 2>&1; then
    expected="$(awk '{print $1}' "$TMP_SHA256")  ${tmp_target}"
    echo "$expected" | sha256sum -c - >/dev/null
  else
    log "Warning: neither shasum nor sha256sum found; skipping checksum verification"
  fi

  rm -f "$TMP_SHA256"
  chmod +x "$tmp_target"
  mv "$tmp_target" "$TARGET"
  INSTALLED_THIS_RUN=1
}

is_go_binary() {
  [[ -x "$TARGET" ]] || return 1
  "$TARGET" version >/dev/null 2>&1
}

ensure_go_binary() {
  if ! is_go_binary; then
    download_binary
  fi
}

should_auto_index() {
  case "${1:-}" in
    query|status)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

main() {
  local subcommand

  ensure_go_binary

  if [[ $# -eq 0 ]]; then
    printf '%s\n' "$TARGET"
    exit 0
  fi

  subcommand="$1"
  shift

  if [[ "$INSTALLED_THIS_RUN" -eq 1 ]] && should_auto_index "$subcommand"; then
    log "Building search index with the Go binary..."
    "$TARGET" index
  fi

  exec "$TARGET" "$subcommand" "$@"
}

main "$@"
