import tempfile
import unittest
from pathlib import Path

from app.rag.vault import VaultRAG


class VaultRefreshTests(unittest.TestCase):
    def test_frontmatter_decodes_json_escaped_quotes(self):
        meta, _ = VaultRAG._split_frontmatter(
            '---\ntitle: "Тест"\norganization: "Общество \\"Нейро\\""\n---\nBody'
        )
        self.assertEqual(meta["title"], "Тест")
        self.assertEqual(meta["organization"], 'Общество "Нейро"')

    def test_load_or_build_rebuilds_when_vault_documents_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / "vault"
            index = root / "index"
            vault.mkdir()
            (vault / "first.md").write_text(
                "---\ntitle: First\n---\n\nFirst document.", encoding="utf-8"
            )
            first = VaultRAG(vault, index)
            first_count = first.load_or_build()

            (vault / "second.md").write_text(
                "---\ntitle: Second\n---\n\nSecond document.", encoding="utf-8"
            )
            second = VaultRAG(vault, index)
            second_count = second.load_or_build()

            self.assertGreater(second_count, first_count)
            self.assertTrue(any(chunk.title == "Second" for chunk in second.chunks))

    def test_full_text_chunks_keep_page_number(self):
        chunks = VaultRAG._chunk_text(
            "# Мигрень\n\n## Страница 1\n\nПервая страница.\n\n"
            "## Страница 2\n\nВторая страница.",
            max_len=80,
        )
        self.assertTrue(any(chunk.startswith("[Страница 1]") for chunk in chunks))
        self.assertTrue(any(chunk.startswith("[Страница 2]") for chunk in chunks))

    def test_hybrid_search_prefers_official_title_and_mkb_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / "vault"
            index = root / "index"
            vault.mkdir()
            (vault / "secondary.md").write_text(
                "---\n"
                "title: \"Общий обзор головной боли\"\n"
                "content_scope: \"reviewed_excerpt\"\n"
                "---\n\nМигрень и другие состояния.",
                encoding="utf-8",
            )
            (vault / "official.md").write_text(
                "---\n"
                "title: \"Мигрень\"\n"
                "mkb_codes: \"G43, G43.0\"\n"
                "content_scope: \"official_full_text_extracted\"\n"
                "---\n\n## Страница 1\n\nОфициальный текст.",
                encoding="utf-8",
            )
            rag = VaultRAG(vault, index)
            rag.build()
            hits = rag.search("Лечение мигрени G43", k=2)
            self.assertEqual(hits[0]["title"], "Мигрень")
            self.assertEqual(
                hits[0]["meta"]["content_scope"], "official_full_text_extracted"
            )


if __name__ == "__main__":
    unittest.main()
