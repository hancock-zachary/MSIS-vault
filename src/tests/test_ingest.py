import json
import pytest
from unittest.mock import patch, MagicMock
from src.ingest import (
    find_unindexed_files, log_indexed, load_log, _course_from_path,
    _source_type_from_path, purge_deleted_files,
)


def test_purge_guard_aborts_and_changes_nothing_when_many_missing(tmp_path):
    # 5 of 5 logged files missing from disk → looks like a reorg, not a
    # deletion. Guard must refuse and leave Chroma + the log untouched.
    log = {str(tmp_path / f"f{i}.pdf"): "done" for i in range(5)}
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(log))
    with patch("src.ingest.get_collection") as mock_get:
        with pytest.raises(SystemExit):
            purge_deleted_files(dict(log), log_path, allow_purge=False)
    mock_get.return_value.delete.assert_not_called()
    assert json.loads(log_path.read_text()) == log  # log untouched


def test_purge_guard_overridden_by_allow_purge(tmp_path):
    import src.ingest as ingest_mod
    orig = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    try:
        log = {str(tmp_path / "IS 6410" / "slides" / f"f{i}.pdf"): "done" for i in range(5)}
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(log))
        col = MagicMock()
        col.get.return_value = {"ids": []}
        with patch("src.ingest.get_collection", return_value=col):
            purge_deleted_files(log, log_path, allow_purge=True)
        assert json.loads(log_path.read_text()) == {}
    finally:
        ingest_mod.RAW_DIR = orig


def test_purge_below_threshold_proceeds_without_flag(tmp_path):
    import src.ingest as ingest_mod
    orig = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    try:
        log = {}
        for i in range(9):
            p = tmp_path / "IS 6410" / "slides" / f"keep{i}.pdf"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            log[str(p)] = "done"
        gone = tmp_path / "IS 6410" / "slides" / "gone.pdf"
        log[str(gone)] = "done"  # 1 of 10 missing = 10% < threshold
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(log))
        col = MagicMock()
        col.get.return_value = {"ids": ["x"]}
        with patch("src.ingest.get_collection", return_value=col):
            purge_deleted_files(log, log_path, allow_purge=False)
        remaining = json.loads(log_path.read_text())
        assert str(gone) not in remaining
        assert len(remaining) == 9
    finally:
        ingest_mod.RAW_DIR = orig


def test_purge_scopes_delete_to_course_and_filename(tmp_path):
    # Same basename in two courses; only the missing course's chunks purged.
    import src.ingest as ingest_mod
    orig = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    try:
        keep = tmp_path / "OSC 6660" / "slides" / "Module 1.pdf"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.touch()
        gone = tmp_path / "IS 6410" / "slides" / "Module 1.pdf"  # missing
        log = {str(keep): "done", str(gone): "done"}
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(log))
        col = MagicMock()
        col.get.return_value = {"ids": []}
        with patch("src.ingest.get_collection", return_value=col):
            purge_deleted_files(log, log_path, allow_purge=True)
        where = col.get.call_args.kwargs["where"]
        assert where == {"$and": [{"filename": "Module 1.pdf"}, {"course": "IS 6410"}]}
    finally:
        ingest_mod.RAW_DIR = orig
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

def test_source_type_from_path_maps_known_subfolders(tmp_path):
    import src.ingest as ingest_mod
    cases = {
        "slides": "slides",
        "readings": "reading",
        "transcripts": "transcript",
        "Assignments": "assignment",
        "additional_files": "unknown",
    }
    original = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    try:
        for folder, expected in cases.items():
            d = tmp_path / "IS 6410" / folder
            d.mkdir(parents=True, exist_ok=True)
            f = d / "x.pdf"
            f.touch()
            assert _source_type_from_path(f) == expected, folder
    finally:
        ingest_mod.RAW_DIR = original


def test_source_type_from_path_no_subfolder_is_unknown(tmp_path):
    import src.ingest as ingest_mod
    original = ingest_mod.RAW_DIR
    ingest_mod.RAW_DIR = tmp_path
    try:
        (tmp_path / "IS 6410").mkdir()
        f = tmp_path / "IS 6410" / "x.pdf"
        f.touch()
        assert _source_type_from_path(f) == "unknown"
        g = tmp_path / "loose.pdf"
        g.touch()
        assert _source_type_from_path(g) == "unknown"
    finally:
        ingest_mod.RAW_DIR = original


def test_is_quality_text_imported_correctly():
    assert callable(is_quality_text)

def test_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.json"
    log = {}
    log_indexed(log, tmp_path / "a.pdf", log_path)
    loaded = load_log(log_path)
    assert str(tmp_path / "a.pdf") in loaded
