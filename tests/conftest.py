from unittest.mock import patch

import pytest

from cc_search.indexer import load_model


@pytest.fixture(scope="session")
def model():
    return load_model()


@pytest.fixture(autouse=True, scope="session")
def _shared_model(model):
    with patch("cc_search.indexer.load_model", return_value=model), \
         patch("cc_search.searcher.load_model", return_value=model):
        yield
