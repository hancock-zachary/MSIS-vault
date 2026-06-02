import pytest
import fitz
from pathlib import Path

@pytest.fixture
def sample_pdf(tmp_path):
    """Returns path to a minimal test PDF with known text content."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Entity-Relationship Diagrams\nAn ERD models data entities and their relationships.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "IS 6410_test.pdf_p1_c0",
            "course": "IS 6410",
            "filename": "test.pdf",
            "page": 1,
            "slide_title": "Entity-Relationship Diagrams",
            "chunk_index": 0,
            "text": "An ERD models data entities and their relationships.",
        }
    ]
