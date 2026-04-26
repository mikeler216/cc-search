import os
import time

import click

from cc_search.indexer import DEFAULT_DB_PATH, DEFAULT_CLAUDE_DIR, Indexer
from cc_search.searcher import Searcher


def _cwd_to_project(cwd: str) -> str:
    return cwd.lstrip("/").replace("-", "/")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
@click.option("--claude-dir", default=DEFAULT_CLAUDE_DIR, help="Path to ~/.claude directory")
@click.option("--full", is_flag=True, help="Force full reindex")
@click.option("--watch", is_flag=True, help="Watch for changes and reindex")
def index(db_path, claude_dir, full, watch):
    indexer = Indexer(db_path=db_path, claude_dir=claude_dir)
    if watch:
        click.echo("Watching for changes... (Ctrl+C to stop)")
        while True:
            indexer.index(full=False)
            time.sleep(30)
    else:
        indexer.index(full=full)
        stats = indexer.db.get_stats()
        click.echo(
            f"Indexed {stats['total_chunks']} chunks from {stats['total_files']} files."
        )
    indexer.db.close()


@cli.command()
@click.argument("query_words", nargs=-1, required=True)
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
@click.option("--top", default=5, help="Number of results to return")
@click.option("--all", "search_all", is_flag=True, help="Search all projects instead of current directory")
@click.option("--project", default=None, help="Filter by project path")
@click.option("--role", default=None, type=click.Choice(["user", "assistant"]), help="Filter by role")
def query(query_words, db_path, top, search_all, project, role):
    query_text = " ".join(query_words)
    if not search_all and project is None:
        project = _cwd_to_project(os.getcwd())
    searcher = Searcher(db_path=db_path)
    results = searcher.search(query_text, top_k=top, project=project, role=role)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        click.echo(f"── Result {i} (score: {score:.2f}) {'─' * 30}")
        click.echo(f"Project: {r['project']}")
        click.echo(f"Role:    {r['role']}")
        click.echo(f"Turn:    {r['turn_index']}")
        click.echo()
        text = r["chunk_text"]
        if len(text) > 300:
            text = text[:300] + "..."
        for line in text.split("\n"):
            click.echo(f"  {line}")
        click.echo()
        click.echo(f"  → {r['resume_command']}")
        click.echo()

    searcher.db.close()


@cli.command()
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
def status(db_path):
    from cc_search.db import SearchDB

    db = SearchDB(db_path)
    stats = db.get_stats()
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    click.echo(f"Chunks:  {stats['total_chunks']}")
    click.echo(f"Files:   {stats['total_files']}")
    click.echo(f"DB size: {db_size / 1024:.1f} KB")
    db.close()
