import unittest

from app.documents.models import DocumentPage
from app.documents.redactor import redact_pages


class DocumentRedactionTests(unittest.TestCase):
    def test_redacts_common_identifiers_without_losing_page(self) -> None:
        pages = (
            DocumentPage(
                number=2,
                text=(
                    "ФИО: Иванов Иван Александрович\n"
                    "Телефон +7 (999) 123-45-67\n"
                    "Email patient@example.org\n"
                    "СНИЛС 123-456-789 00\n"
                    "Диагноз: синтетический пример"
                ),
            ),
        )
        redacted, report = redact_pages(pages)
        self.assertEqual(redacted[0].number, 2)
        self.assertNotIn("Иванов", redacted[0].text)
        self.assertNotIn("patient@example.org", redacted[0].text)
        self.assertNotIn("123-456-789", redacted[0].text)
        self.assertIn("Диагноз: синтетический пример", redacted[0].text)
        self.assertGreaterEqual(report.total_replacements, 4)
        self.assertFalse(report.review_required)

    def test_marks_unhandled_pii_label_for_review(self) -> None:
        pages = (
            DocumentPage(number=1, text="Паспорт: не предъявлен\nЗаключение: тест"),
        )
        _, report = redact_pages(pages)
        self.assertTrue(report.review_required)
        self.assertTrue(report.warnings)


if __name__ == "__main__":
    unittest.main()
