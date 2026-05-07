#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/scripts/run-cc-search.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_eq() {
  local got="$1"
  local want="$2"
  local message="$3"
  if [[ "$got" != "$want" ]]; then
    fail "${message}: got=${got@Q} want=${want@Q}"
  fi
}

assert_file_contains_line() {
  local file="$1"
  local line="$2"
  if ! grep -Fxq "$line" "$file"; then
    fail "expected ${file} to contain line ${line@Q}"
  fi
}

test_installs_go_binary_even_when_legacy_cc_search_is_on_path() {
  local tmpdir home_dir legacy_dir fake_go legacy_log go_log query_output

  tmpdir="$(mktemp -d /tmp/cc-search-test.XXXXXX)"
  trap 'rm -rf "$tmpdir"' RETURN

  home_dir="${tmpdir}/home"
  legacy_dir="${tmpdir}/legacy/bin"
  legacy_log="${tmpdir}/legacy.log"
  go_log="${tmpdir}/go.log"
  fake_go="${tmpdir}/cc-search-go"

  mkdir -p "${home_dir}" "${legacy_dir}"

  cat >"${legacy_dir}/cc-search" <<EOF
#!/usr/bin/env bash
echo "legacy \$*" >>"${legacy_log}"
echo "legacy python cc-search"
exit 0
EOF
  chmod +x "${legacy_dir}/cc-search"

  cat >"${fake_go}" <<EOF
#!/usr/bin/env bash
echo "\$*" >>"${go_log}"
case "\$1" in
  version)
    echo "1.0.1"
    ;;
  index)
    exit 0
    ;;
  query)
    shift
    printf 'go query:%s\n' " \$*"
    ;;
  *)
    echo "unexpected: \$*" >&2
    exit 1
    ;;
esac
EOF
  chmod +x "${fake_go}"

  query_output="$(
    HOME="${home_dir}" \
    PATH="${legacy_dir}:$PATH" \
    CC_SEARCH_TEST_BINARY_SOURCE="${fake_go}" \
    "${SCRIPT}" query migration-test 2>"${tmpdir}/stderr.log"
  )"

  assert_eq "${query_output}" "go query: migration-test" "query output should come from the Go binary"
  [[ -x "${home_dir}/.local/bin/cc-search" ]] || fail "expected Go binary install at ${home_dir}/.local/bin/cc-search"
  [[ ! -f "${legacy_log}" ]] || fail "legacy PATH binary should not be invoked"
  assert_file_contains_line "${go_log}" "index"
  assert_file_contains_line "${go_log}" "query migration-test"
}

test_rejects_unsupported_platform() {
  local tmpdir home_dir stderr_file

  tmpdir="$(mktemp -d /tmp/cc-search-test.XXXXXX)"
  trap 'rm -rf "$tmpdir"' RETURN

  home_dir="${tmpdir}/home"
  stderr_file="${tmpdir}/stderr.log"
  mkdir -p "${home_dir}"

  if HOME="${home_dir}" CC_SEARCH_TEST_OS="darwin" CC_SEARCH_TEST_ARCH="amd64" "${SCRIPT}" query test 2>"${stderr_file}"; then
    fail "expected unsupported platform invocation to fail"
  fi

  if ! grep -Fq "unsupported platform darwin/amd64" "${stderr_file}"; then
    fail "expected unsupported platform message in stderr"
  fi
}

test_installs_go_binary_even_when_legacy_cc_search_is_on_path
test_rejects_unsupported_platform
echo "PASS"
