import json
from unittest.mock import patch, MagicMock
from brain.ingest import find_unindexed_files, log_indexed, load_log, _is_quality_chunk

def test_find_unindexed_files_finds_all_types(tmp_path):
    (tmp_path / "course").mkdir()
    files = {
        tmp_path / "course" / "slides.pdf",
        tmp_path / "course" / "notes.md",
        tmp_path / "course" / "reading.txt",
        tmp_path / "course" / "report.docx",
    }
    for f in files:
        f.touch()
    log = {}
    result = set(find_unindexed_files(tmp_path, log))
    assert files == result

def test_already_indexed_file_excluded(tmp_path):
    pdf1 = tmp_path / "slides.pdf"
    pdf1.touch()
    log = {str(pdf1): "done"}
    result = find_unindexed_files(tmp_path, log)
    assert pdf1 not in result

def test_unsupported_extension_excluded(tmp_path):
    pptx = tmp_path / "deck.pptx"
    pptx.touch()
    result = find_unindexed_files(tmp_path, {})
    assert pptx not in result

def test_quality_chunk_accepts_normal_text():
    text = "The five Scrum events are Sprint Planning, Daily Scrum, Sprint Review, Sprint Retrospective, and the Sprint itself."
    assert _is_quality_chunk(text) is True

def test_quality_chunk_rejects_garbled_image_text():
    garbled = "E 8 c I M O O 75 2 1 S xa S e U n E fl o ij ri O CM o COrAPLICATED THAT NO ONE KNOWS WHO OOES WHAT"
    assert _is_quality_chunk(garbled) is False

def test_quality_chunk_rejects_too_short():
    assert _is_quality_chunk("too short") is False

def test_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.json"
    log = {}
    log_indexed(log, tmp_path / "a.pdf", log_path)
    loaded = load_log(log_path)
    assert str(tmp_path / "a.pdf") in loaded
