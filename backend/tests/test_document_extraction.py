import unittest
from io import BytesIO

from app.documents.extractor import (
    DocumentValidationError,
    extract_pdf,
    validate_upload_metadata,
)
from pypdf import PdfWriter
from pypdf.actions import JavaScript


def make_text_pdf(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            commands.append("0 -18 Td")
        commands.append(f"({escaped}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


class DocumentExtractionTests(unittest.TestCase):
    def test_extracts_text_with_page_number(self) -> None:
        data = make_text_pdf(["Diagnosis: synthetic case", "Glucose 5.4 mmol/L"])
        result = extract_pdf(data, max_pages=5, max_chars=10_000)
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].number, 1)
        self.assertIn("synthetic case", result.pages[0].text)
        self.assertEqual(len(result.sha256), 64)

    def test_rejects_non_pdf_signature(self) -> None:
        with self.assertRaisesRegex(DocumentValidationError, "Сигнатура"):
            extract_pdf(b"not-a-pdf", max_pages=5, max_chars=10_000)

    def test_rejects_active_javascript(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_open_action(JavaScript("app.alert('synthetic')"))
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(DocumentValidationError, "активное содержимое"):
            extract_pdf(output.getvalue(), max_pages=5, max_chars=10_000)

    def test_rejects_too_many_pages(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_blank_page(width=100, height=100)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(DocumentValidationError, "больше допустимых"):
            extract_pdf(output.getvalue(), max_pages=1, max_chars=10_000)

    def test_rejects_encrypted_pdf(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("synthetic-password")
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(DocumentValidationError, "Защищённые паролем"):
            extract_pdf(output.getvalue(), max_pages=5, max_chars=10_000)

    def test_rejects_pdf_without_text_layer(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(DocumentValidationError, "нужен OCR"):
            extract_pdf(output.getvalue(), max_pages=5, max_chars=10_000)

    def test_validates_extension_and_content_type(self) -> None:
        validate_upload_metadata("synthetic.PDF", "application/pdf")
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata("synthetic.txt", "application/pdf")
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata("synthetic.pdf", "image/png")


if __name__ == "__main__":
    unittest.main()
