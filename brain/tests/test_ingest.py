import json
from unittest.mock import patch, MagicMock
from brain.ingest import find_unindexed_pdfs, log_indexed, load_log

def test_find_unindexed_pdfs(tmp_path):
    (tmp_path / "course").mkdir()
    pdf1 = tmp_path / "course" / "slides.pdf"
    pdf1.touch()
    log = {}
    result = find_unindexed_pdfs(tmp_path, log)
    assert pdf1 in result

def test_already_indexed_pdf_excluded(tmp_path):
    pdf1 = tmp_path / "slides.pdf"
    pdf1.touch()
    log = {str(pdf1): "done"}
    result = find_unindexed_pdfs(tmp_path, log)
    assert pdf1 not in result

def test_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.json"
    log = {}
    log_indexed(log, tmp_path / "a.pdf", log_path)
    loaded = load_log(log_path)
    assert str(tmp_path / "a.pdf") in loaded
