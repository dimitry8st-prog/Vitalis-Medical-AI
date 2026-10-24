import unittest

from app.documents.models import (
    DocumentAnalysis,
    DocumentPage,
    PiiReport,
    SecurityReport,
)
from app.documents.session_store import DocumentSessionStore


class DocumentSessionStoreTests(unittest.TestCase):
    def _create(self, store: DocumentSessionStore):
        return store.create(
            pages=(DocumentPage(number=1, text="synthetic document text"),),
            analysis=DocumentAnalysis(summary="synthetic"),
            pii=PiiReport(),
            security=SecurityReport(),
            llm_mode="mock",
        )

    def test_session_expires_and_can_be_deleted(self) -> None:
        now = [100.0]
        store = DocumentSessionStore(
            ttl_seconds=60, max_sessions=2, clock=lambda: now[0]
        )
        item = self._create(store)
        self.assertIsNotNone(store.get(item.session_id))
        self.assertTrue(store.delete(item.session_id))
        self.assertIsNone(store.get(item.session_id))

        item = self._create(store)
        now[0] = 161.0
        self.assertIsNone(store.get(item.session_id))

    def test_oldest_session_is_evicted_at_capacity(self) -> None:
        now = [100.0]
        store = DocumentSessionStore(
            ttl_seconds=60, max_sessions=1, clock=lambda: now[0]
        )
        first = self._create(store)
        now[0] = 101.0
        second = self._create(store)
        self.assertIsNone(store.get(first.session_id))
        self.assertIsNotNone(store.get(second.session_id))


if __name__ == "__main__":
    unittest.main()
