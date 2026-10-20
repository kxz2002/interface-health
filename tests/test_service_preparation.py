import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from src.service import sn_preparation
from src.service.preparation import checked_path, file_digest
from src.service.sn_preparation import iter_json_array


def test_checked_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="outside dataset"):
        checked_path(tmp_path, "../outside")


def test_checked_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-preparation.txt"
    outside.write_text("external")
    link = tmp_path / "link"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside dataset"):
        checked_path(tmp_path, "link")


def test_digest_matches_content(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"normal observation")
    assert file_digest(source) == hashlib.sha256(b"normal observation").hexdigest()


@pytest.mark.parametrize("value", ["[{},]", "[{}", "[{}] garbage", "[{} {}]"])
def test_stream_rejects_invalid_json(value):
    with pytest.raises(ValueError):
        list(iter_json_array(io.StringIO(value)))


def test_stream_accepts_empty_and_nested_arrays():
    assert list(iter_json_array(io.StringIO("[]"))) == []
    assert list(iter_json_array(io.StringIO('[{"spans": [1, 2]}, {}]'))) == [{"spans": [1, 2]}, {}]


def test_stream_handles_large_objects_across_chunks():
    value = {"message": "a" * 150000}
    assert list(iter_json_array(io.StringIO(json.dumps([value, {}])))) == [value, {}]


def test_sn_archive_retains_unknown_error_and_checks_overlap(tmp_path, monkeypatch):
    archive = tmp_path / "sn.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        for index in range(2):
            run_id = f"SN.run{index}"
            content = {
                f"{run_id}/metrics/service.csv": b"timestamp,cpu_usage_total\n100,1\n101,2\n",
                f"{run_id}/logs.json": json.dumps(
                    {"service": [f"timestamp-{index} message"]}
                ).encode(),
                f"{run_id}/spans.json": json.dumps(
                    [
                        {
                            "traceID": "shared",
                            "processes": {"p": {"serviceName": "service"}},
                            "spans": [{"processID": "p", "startTime": 100000000, "tags": []}],
                        }
                    ]
                ).encode(),
            }
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:xz") as inner:
                for name, payload in content.items():
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    inner.addfile(member, io.BytesIO(payload))
            outer.writestr(f"SN Dataset/no fault/{run_id}.tar.xz", buffer.getvalue())
            outer.writestr(
                f"SN Dataset/no fault/SN.fault-run{index}.json",
                json.dumps({"start": 100 + index * 100, "end": 110 + index * 100, "faults": []}),
            )
    monkeypatch.setattr(sn_preparation, "SN_MD5", file_digest(archive, "md5"))
    result = sn_preparation.inspect_sn(archive)
    assert result["normal_overlap"][0]["trace_id_intersection"] == 1
    assert result["normal_overlap"][0]["trace_id_in_both_run_bounds"] == 0
    assert result["runs"]["SN.run0"]["traces"]["tags"] == {}
    assert result["runs"]["SN.run0"]["metrics"]["service"]["rows"] == 2
