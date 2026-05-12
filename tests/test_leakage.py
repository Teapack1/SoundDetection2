"""Tests for train/test leakage detection and the batch-evaluate pre-flight."""

from __future__ import annotations

from pathlib import Path

import pytest

from sed.train.leakage import (
    find_leakage,
    normalize_clip_path,
    read_clip_rows,
    training_index,
)
from sed.train.test_compat import (
    ClassMapping,
    check_run,
    check_runs,
    parse_class_mapping,
    suggest_class_mapping,
)

HEADER = "path,class_name,session_id,sample_rate,duration_s,is_target,origin\n"


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{p},{c},{s},44100,0.4,1,sounds\n" for p, c, s in rows)
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def _run_dir(tmp_path: Path, name: str, train: list[tuple[str, str, str]], *, ckpt=True) -> Path:
    d = tmp_path / "models" / name
    _write_csv(d / "dataset" / "train.csv", train)
    _write_csv(d / "dataset" / "val.csv", [])
    if ckpt:
        (d / "checkpoint_best_epoch-001.ckpt").write_bytes(b"stub")
    return d


def test_identical_clip_in_training_data_is_clip_leakage(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    shared = str(lib / "sess_0001.wav")
    run = _run_dir(tmp_path, "r1", [(shared, "cheers", "sess")])

    test_rows = [
        {"path": shared, "class_name": "cheers", "session_id": "sess"},
        {"path": str(lib / "other_0001.wav"), "class_name": "cheers", "session_id": "other"},
    ]
    leak = find_leakage(test_rows, training_index(run, "r1"))

    assert leak["clip_leak_count"] == 1
    assert leak["group_leak_count"] == 0
    assert leak["leaked_paths"] == [shared]
    assert leak["leak_fraction"] == pytest.approx(0.5)


def test_a_different_clip_from_the_same_recording_is_group_leakage(tmp_path):
    """The case that silently inflates scores: not the same file, but a clip cut
    from the same recording seconds away — effectively the same sample."""
    lib = tmp_path / "lib"
    lib.mkdir()
    trained_on = str(lib / "sess_0001.wav")
    sibling = str(lib / "sess_0002.wav")
    run = _run_dir(tmp_path, "r1", [(trained_on, "cheers", "sess")])

    leak = find_leakage(
        [{"path": sibling, "class_name": "cheers", "session_id": "sess"}],
        training_index(run, "r1"),
    )

    assert leak["clip_leak_count"] == 0
    assert leak["group_leak_count"] == 1
    assert leak["leaked_paths"] == [sibling]


def test_clip_and_group_leaks_are_disjoint_so_the_counts_sum(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    exact = str(lib / "sess_0001.wav")
    sibling = str(lib / "sess_0002.wav")
    clean = str(lib / "elsewhere_0001.wav")
    run = _run_dir(tmp_path, "r1", [(exact, "cheers", "sess")])

    leak = find_leakage(
        [
            {"path": exact, "class_name": "cheers", "session_id": "sess"},
            {"path": sibling, "class_name": "cheers", "session_id": "sess"},
            {"path": clean, "class_name": "cheers", "session_id": "elsewhere"},
        ],
        training_index(run, "r1"),
    )

    assert (leak["clip_leak_count"], leak["group_leak_count"]) == (1, 1)
    assert leak["leak_count"] == 2
    assert clean not in leak["leaked_paths"]


def test_a_run_without_a_dataset_copy_reports_unavailable_not_clean(tmp_path):
    """Absence of evidence is not evidence of absence — an unverifiable run must
    not read as verified-clean."""
    empty = tmp_path / "models" / "r0"
    empty.mkdir(parents=True)
    index = training_index(empty, "r0")

    assert index.available is False
    assert find_leakage([{"path": "/x/a.wav", "class_name": "c"}], index)["available"] is False


def test_val_split_counts_as_training_data_but_the_runs_own_test_does_not(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    in_val = str(lib / "vsess_0001.wav")
    in_own_test = str(lib / "tsess_0001.wav")
    run = tmp_path / "models" / "r1"
    _write_csv(run / "dataset" / "train.csv", [])
    _write_csv(run / "dataset" / "val.csv", [(in_val, "cheers", "vsess")])
    _write_csv(run / "dataset" / "test.csv", [(in_own_test, "cheers", "tsess")])

    index = training_index(run, "r1")
    leak = find_leakage(
        [
            {"path": in_val, "class_name": "cheers", "session_id": "vsess"},
            {"path": in_own_test, "class_name": "cheers", "session_id": "tsess"},
        ],
        index,
    )
    assert leak["leaked_paths"] == [in_val]


def test_normalize_clip_path_matches_equivalent_spellings(tmp_path):
    a = normalize_clip_path(str(tmp_path / "x" / ".." / "a.wav"))
    b = normalize_clip_path(str(tmp_path / "a.wav"))
    assert a == b


def test_read_clip_rows_tolerates_a_missing_file(tmp_path):
    assert read_clip_rows(tmp_path / "nope.csv") == []


# ---- pre-flight ------------------------------------------------------------


def test_check_run_reports_leakage_with_a_removal_remedy(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    shared = str(lib / "sess_0001.wav")
    run = _run_dir(tmp_path, "r1", [(shared, "cheers", "sess")])
    test_csv = _write_csv(tmp_path / "test.csv", [(shared, "cheers", "sess")])

    result = check_run(read_clip_rows(test_csv), run, "r1", ["_negative", "cheers"])

    assert result["verdict"] == "fixable"
    leak = next(f for f in result["findings"] if f["code"] == "leakage")
    assert leak["remedy"]["kind"] == "drop_leaked_clips"
    assert leak["remedy"]["paths"] == [shared]


def test_check_run_flags_a_class_the_run_cannot_score(tmp_path):
    """The state the pretraining project is in: runs on `cheers`, test on
    `loud/medium/quiet`."""
    run = _run_dir(tmp_path, "r1", [(str(tmp_path / "a.wav"), "cheers", "s1")])
    test_csv = _write_csv(tmp_path / "test.csv", [(str(tmp_path / "b.wav"), "loud", "s2")])

    result = check_run(read_clip_rows(test_csv), run, "r1", ["_negative", "cheers"])

    finding = next(f for f in result["findings"] if f["code"] == "class_space")
    assert finding["classes"] == ["loud"]
    assert finding["remedy"]["kind"] == "class_mapping"
    assert finding["remedy"]["suggestion"] == {}   # "loud" -> "cheers" is not inferable


def test_a_class_mapping_makes_an_orphaned_run_scorable(tmp_path):
    """Folding a refined taxonomy back onto the run's coarser one revives it."""
    run = _run_dir(tmp_path, "r1", [(str(tmp_path / "a.wav"), "cheers", "s1")])
    test_csv = _write_csv(
        tmp_path / "test.csv",
        [
            (str(tmp_path / "b.wav"), "loud", "s2"),
            (str(tmp_path / "c.wav"), "quiet", "s3"),
            (str(tmp_path / "d.wav"), "_negative", "s4"),
        ],
    )
    rows = read_clip_rows(test_csv)
    classes = ["_negative", "cheers"]

    # Unmapped the run cannot be scored — but it is fixable, not a dead end.
    before = check_run(rows, run, "r1", classes)
    assert before["verdict"] == "fixable"
    assert next(f for f in before["findings"] if f["code"] == "class_space")["classes"] == [
        "loud", "quiet",
    ]

    mapping = parse_class_mapping({"loud": "cheers", "quiet": "cheers"})
    after = check_run(rows, run, "r1", classes, mapping=mapping)
    assert after["verdict"] == "ok"
    assert not any(f["code"] == "class_space" for f in after["findings"])


def test_suggest_class_mapping_only_proposes_unambiguous_merges():
    # `cheers_quiet` is plainly a refinement of `cheers`.
    assert suggest_class_mapping(
        {"cheers_quiet", "cheers_loud"}, ["_negative", "cheers"]
    ) == {"cheers_quiet": "cheers", "cheers_loud": "cheers"}
    # An unrelated name must not be guessed at — a wrong merge changes what is measured.
    assert suggest_class_mapping({"whistle"}, ["_negative", "cheers"]) == {}


def test_class_mapping_only_merges_and_never_invents_a_distinction():
    mapping = ClassMapping({"loud": "cheers", "quiet": "cheers"})
    rows = [{"path": "a", "class_name": "loud"}, {"path": "b", "class_name": "_negative"}]
    assert [r["class_name"] for r in mapping.map_rows(rows)] == ["cheers", "_negative"]
    assert ClassMapping({}).is_empty is True


def test_check_run_warns_when_the_test_does_not_cover_a_trained_class(tmp_path):
    run = _run_dir(tmp_path, "r1", [(str(tmp_path / "a.wav"), "cheers", "s1")])
    test_csv = _write_csv(tmp_path / "test.csv", [(str(tmp_path / "b.wav"), "cheers", "s2")])

    result = check_run(read_clip_rows(test_csv), run, "r1", ["_negative", "cheers", "clap"])

    assert result["verdict"] == "ok", "missing coverage is a warning, not a blocker"
    assert next(f for f in result["findings"] if f["code"] == "class_coverage")["classes"] == ["clap"]


def test_a_run_without_a_checkpoint_is_incompatible(tmp_path):
    run = _run_dir(tmp_path, "r1", [(str(tmp_path / "a.wav"), "cheers", "s1")], ckpt=False)
    test_csv = _write_csv(tmp_path / "test.csv", [(str(tmp_path / "b.wav"), "cheers", "s2")])

    result = check_run(read_clip_rows(test_csv), run, "r1", ["_negative", "cheers"])
    assert result["verdict"] == "incompatible"
    assert any(f["code"] == "no_checkpoint" for f in result["findings"])


def test_check_runs_reports_the_union_of_leaked_clips_across_the_batch(tmp_path):
    """A shared benchmark must be clean for every model it ranks, so the remedy
    drops the union rather than a per-run subset."""
    lib = tmp_path / "lib"
    lib.mkdir()
    a, b = str(lib / "asess_0001.wav"), str(lib / "bsess_0001.wav")
    run_a = _run_dir(tmp_path, "ra", [(a, "cheers", "asess")])
    run_b = _run_dir(tmp_path, "rb", [(b, "cheers", "bsess")])
    test_csv = _write_csv(tmp_path / "test.csv", [(a, "cheers", "asess"), (b, "cheers", "bsess")])

    report = check_runs(test_csv, [
        {"id": "ra", "model_dir": run_a, "classes": ["_negative", "cheers"]},
        {"id": "rb", "model_dir": run_b, "classes": ["_negative", "cheers"]},
    ])

    assert report["counts"]["fixable"] == 2
    assert report["all_clear"] is False
    assert sorted(report["leaked_paths_union"]) == sorted([a, b])


def test_check_runs_is_clean_for_a_disjoint_test_set(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    run = _run_dir(tmp_path, "r1", [(str(lib / "trainsess_0001.wav"), "cheers", "trainsess")])
    test_csv = _write_csv(
        tmp_path / "test.csv",
        [(str(lib / "heldout_0001.wav"), "cheers", "heldout"),
         (str(lib / "negsess_0001.wav"), "_negative", "negsess")],
    )

    report = check_runs(test_csv, [
        {"id": "r1", "model_dir": run, "classes": ["_negative", "cheers"]},
    ])
    assert report["all_clear"] is True
    assert report["counts"] == {"ok": 1, "fixable": 0, "incompatible": 0}


def test_leakage_and_compat_stay_torch_free():
    import subprocess
    import sys

    probe = (
        "import sed.train.leakage, sed.train.test_compat, sys; print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
