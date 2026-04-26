import json
import os
import tempfile

from click.testing import CliRunner

from cc_search.cli import cli


def _setup_test_env(tmpdir):
    claude_dir = os.path.join(tmpdir, "claude")
    projects_dir = os.path.join(claude_dir, "projects", "-Users-test-proj")
    os.makedirs(projects_dir)
    db_path = os.path.join(tmpdir, "test.db")

    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "How do I configure JWT auth?"},
            "sessionId": "sess-abc",
            "timestamp": 1000,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Set up JWT middleware like this."}],
            },
            "sessionId": "sess-abc",
            "timestamp": 1001,
        },
    ]
    jsonl_path = os.path.join(projects_dir, "sess-abc.jsonl")
    with open(jsonl_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return claude_dir, db_path


def test_index_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        result = runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        assert result.exit_code == 0
        assert "Indexed" in result.output


def test_query_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        result = runner.invoke(
            cli, ["query", "JWT auth", "--db-path", db_path, "--top", "3", "--all"]
        )
        assert result.exit_code == 0
        assert "claude --resume" in result.output


def test_status_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        result = runner.invoke(cli, ["status", "--db-path", db_path])
        assert result.exit_code == 0
        assert "chunks" in result.output.lower() or "Chunks" in result.output


def test_query_empty_db():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "empty.db")
        result = runner.invoke(cli, ["query", "hello", "--db-path", db_path])
        assert result.exit_code == 0
        assert "No results" in result.output or "no results" in result.output.lower()
