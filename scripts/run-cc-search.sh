#!/usr/bin/env bash
set -euo pipefail

REPO="${CC_SEARCH_REPO:-mikeler216/cc-search}"
INSTALL_DIR="${CC_SEARCH_INSTALL_DIR:-$HOME/.local/bin}"
TARGET="${INSTALL_DIR}/cc-search"
TMP_SHA256="${CC_SEARCH_TMP_SHA256:-/tmp/cc-search.sha256}"
INSTALLED_THIS_RUN=0
LATEST_RELEASE_URL="https://github.com/${REPO}/releases/latest"
LATEST_RELEASE_API_URL="https://api.github.com/repos/${REPO}/releases/latest"

log() {
  echo "$*" >&2
}

detect_os() {
  if [[ -n "${CC_SEARCH_TEST_OS:-}" ]]; then
    printf '%s\n' "${CC_SEARCH_TEST_OS}"
    return
  fi
  uname -s | tr '[:upper:]' '[:lower:]'
}

detect_arch() {
  local arch
  if [[ -n "${CC_SEARCH_TEST_ARCH:-}" ]]; then
    arch="${CC_SEARCH_TEST_ARCH}"
  else
    arch="$(uname -m)"
  fi
  case "$arch" in
    x86_64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) echo "$arch" ;;
  esac
}

platform_supported() {
  case "${1}/${2}" in
    linux/amd64|darwin/arm64)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

unsupported_platform_message() {
  printf 'unsupported platform %s/%s; supported release binaries: linux/amd64 and darwin/arm64\n' "$1" "$2"
}

fetch_latest_tag() {
  local body final_url tag
  if command -v curl >/dev/null 2>&1; then
    final_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "$LATEST_RELEASE_URL" 2>/dev/null || true)"
    tag="${final_url##*/}"
    if [[ "$tag" == v* ]]; then
      printf '%s\n' "$tag"
      return
    fi
    body="$(curl -fsSL "$LATEST_RELEASE_API_URL" 2>/dev/null || true)"
  elif command -v wget >/dev/null 2>&1; then
    body="$(wget -qO- "$LATEST_RELEASE_API_URL" 2>/dev/null || true)"
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

  os="$(detect_os)"
  arch="$(detect_arch)"
  if ! platform_supported "$os" "$arch"; then
    log "Error: $(unsupported_platform_message "$os" "$arch")"
    exit 1
  fi

  latest_tag="$(fetch_latest_tag)"
  if [[ -z "$latest_tag" ]]; then
    log "Error: could not resolve the latest GitHub release tag"
    exit 1
  fi

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
