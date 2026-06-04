import json
from unittest.mock import patch, MagicMock
from src.ingest import find_unindexed_files, log_indexed, load_log, _course_from_path
from src.chunk import is_quality_text
from src.config import RAW_DIR

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

def test_course_from_path_uses_first_subfolder(tmp_path):
    (tmp_path / "IS 6410" / "slides").mkdir(parents=True)
    f = tmp_path / "IS 6410" / "slides" / "week1.pdf"
    f.touch()
    # patch RAW_DIR so _course_from_path resolves relative to tmp_path
    import src.ingest as ingest_mod
    original = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    assert _course_from_path(f) == "IS 6410"
    ingest_mod.RAW_DIR = original

def test_course_from_path_direct_file_is_general(tmp_path):
    f = tmp_path / "random.pdf"
    f.touch()
    import src.ingest as ingest_mod
    original = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    assert _course_from_path(f) == "General"
    ingest_mod.RAW_DIR = original

def test_is_quality_text_imported_correctly():
    assert callable(is_quality_text)

def test_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.json"
    log = {}
    log_indexed(log, tmp_path / "a.pdf", log_path)
    loaded = load_log(log_path)
    assert str(tmp_path / "a.pdf") in loaded
