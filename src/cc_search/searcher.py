from typing import Any

from sentence_transformers import SentenceTransformer

from cc_search.db import SearchDB
from cc_search.indexer import DEFAULT_DB_PATH, MODEL_NAME


class Searcher:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.db = SearchDB(db_path)
        self._model = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.model.encode(query).tolist()
        results = self.db.search(
            query_embedding, top_k=top_k, project=project, role=role
        )
        for r in results:
            r["resume_command"] = f"claude --resume {r['session_id']}"
        return results
