"""M3 WLASL archive tests; no real video decoding or copying."""

import json

import numpy as np

from scripts import build_dataset_module as builder


def make_archive(tmp_path):
    root = tmp_path / "archive"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "00001.mp4").write_bytes(b"video")
    (root / "videos" / "00003.mp4").write_bytes(b"video")
    (root / "wlasl_class_list.txt").write_text("0\tbook\n7\tdrink\n", encoding="utf-8")
    (root / "nslt_100.json").write_text(json.dumps({
        "00001": {"subset": "train", "action": [7, 2, 9]},
        "00002": {"subset": "val", "action": [0, 1, 4]},
        "00003": {"subset": "test", "action": [0, 3, 8]},
    }), encoding="utf-8")
    (root / "WLASL_v0.3.json").write_text(json.dumps([
        {"gloss": "book", "instances": [{"video_id": "00001", "signer_id": 12}]},
        {"gloss": "drink", "instances": [{"video_id": "00003", "signer_id": 13}]},
    ]), encoding="utf-8")
    return root


def fake_extractor(path, **kwargs):
    return np.zeros((3, *builder.FEATURE_SHAPE), dtype=np.float32)


def test_manifest_filters_missing_and_remaps_sparse_ids(tmp_path):
    root = make_archive(tmp_path)
    records = builder.build_wlasl_manifest(root, "100", subset="all")
    assert [r["video_id"] for r in records] == ["00001", "00003"]
    assert [r["original_class_id"] for r in records] == [7, 0]
    assert [r["class_id"] for r in records] == [1, 0]
    assert records[0]["gloss"] == "drink"
    assert records[0]["frame_start"] == 2 and records[0]["frame_end"] == 9
    assert {r["signer_id"] for r in records} == {12, 13}


def test_manifest_subset_and_max_samples(tmp_path):
    root = make_archive(tmp_path)
    records = builder.build_wlasl_manifest(root, "100", subset="train", max_samples=1)
    assert len(records) == 1
    assert records[0]["subset"] == "train"


def test_archive_dataset_keeps_videos_in_place_and_writes_manifest(tmp_path):
    root = make_archive(tmp_path)
    output = tmp_path / "out" / "dataset.npz"
    X, y, metadata = builder.build_wlasl_dataset(root, output, benchmark="100", extractor=fake_extractor)
    assert X.shape == (2, 60, 110, 3)
    assert y.tolist() == [1, 0]
    assert metadata["per_label_counts"] == {"book": 1, "drink": 1}
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert all("video_path" not in row for row in manifest)
    assert {row["video_id"] for row in manifest} == {"00001", "00003"}
    assert (root / "videos" / "00001.mp4").is_file()
