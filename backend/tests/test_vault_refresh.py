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


if __name__ == "__main__":
    unittest.main()
