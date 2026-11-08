import tempfile
import unittest
from pathlib import Path

from app.rag.vault import VaultRAG


def full_guideline(title: str, body: str, *, mkb_codes: str = "") -> str:
    return (
        "---\n"
        f'title: "{title}"\n'
        f'mkb_codes: "{mkb_codes}"\n'
        'document_type: "clinical_guideline_full_text"\n'
        'content_scope: "official_full_text_extracted"\n'
        "full_text_checked: true\n"
        "---\n\n"
        f"{body}"
    )


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
                full_guideline("First", "## Страница 1\n\nFirst document."),
                encoding="utf-8",
            )
            first = VaultRAG(vault, index)
            first_count = first.load_or_build()

            (vault / "second.md").write_text(
                full_guideline("Second", "## Страница 1\n\nSecond document."),
                encoding="utf-8",
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
                "document_type: \"reviewed_lifeos_article\"\n"
                "content_scope: \"reviewed_excerpt\"\n"
                "---\n\nМигрень и другие состояния.",
                encoding="utf-8",
            )
            (vault / "official.md").write_text(
                full_guideline(
                    "Мигрень",
                    "## Страница 1\n\nДля купирования приступа рассматриваются "
                    "триптаны. Профилактическая терапия выбирается по частоте "
                    "приступов.",
                    mkb_codes="G43, G43.0",
                ),
                encoding="utf-8",
            )
            (vault / "epilepsy.md").write_text(
                full_guideline(
                    "Эпилепсия",
                    "## Страница 1\n\nДиагностика приступов включает оценку "
                    "анамнеза и электроэнцефалографию.",
                    mkb_codes="G40",
                ),
                encoding="utf-8",
            )
            rag = VaultRAG(vault, index)
            rag.build()
            hits = rag.search("Лечение мигрени G43", k=2)
            self.assertEqual(hits[0]["title"], "Мигрень")
            self.assertEqual(
                hits[0]["meta"]["content_scope"], "official_full_text_extracted"
            )
            free_query_hits = rag.search(
                "чем купировать приступ и когда нужна профилактическая терапия",
                k=1,
            )
            self.assertEqual(free_query_hits[0]["title"], "Мигрень")

    def test_non_guideline_files_are_preserved_but_never_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault = root / "vault"
            index = root / "index"
            vault.mkdir()
            article = vault / "lifeos-article.md"
            article.write_text(
                "---\n"
                'title: "Статья Life-OS"\n'
                'document_type: "reviewed_lifeos_article"\n'
                'content_scope: "reviewed_excerpt"\n'
                "---\n\nНаучно-популярный материал.",
                encoding="utf-8",
            )

            rag = VaultRAG(vault, index)
            self.assertEqual(rag.build(), 0)
            self.assertTrue(article.exists())
            self.assertEqual(rag.indexed_files, 0)
            self.assertEqual(rag.excluded_files, 1)
            self.assertEqual(rag.search("Статья Life-OS", k=5), [])


if __name__ == "__main__":
    unittest.main()
