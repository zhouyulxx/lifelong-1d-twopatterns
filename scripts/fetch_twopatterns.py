#!/usr/bin/env python3
"""Fetch the original UCR TwoPatterns TRAIN/TEST splits, without generating fake samples."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

URLS = [
    "https://www.timeseriesclassification.com/aeon-toolkit/TwoPatterns.zip",
    "https://timeseriesclassification.com/aeon-toolkit/TwoPatterns.zip",
]
EXPECTED = {"TwoPatterns_TRAIN.ts": 1000, "TwoPatterns_TEST.ts": 4000}


def validate_ts(content: bytes, expected_count: int) -> None:
    text = content.decode("utf-8-sig")
    lines = [s.strip() for s in text.splitlines()]
    data_idx = next((i for i, s in enumerate(lines) if s.lower() == "@data"), None)
    if data_idx is None:
        raise ValueError("Missing @data marker")
    records = [s for s in lines[data_idx + 1:] if s and not s.startswith("#")]
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, got {len(records)}")
    labels = set()
    for line in records:
        sequence, label = line.rsplit(":", 1)
        values = sequence.split(",")
        if len(values) != 128:
            raise ValueError(f"Expected length 128, got {len(values)}")
        float(values[0])
        float(values[-1])
        labels.add(label.strip())
    if len(labels) != 4:
        raise ValueError(f"Expected 4 classes, got {sorted(labels)}")


def download(destination=Path("data/TwoPatterns")):
    last_error = None
    for url in URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Lifelong1DResearch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                archive = response.read()
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                found = {}
                for name, expected in EXPECTED.items():
                    matches = [item for item in z.namelist()
                               if item.replace("\\", "/").split("/")[-1] == name]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one {name}, got {matches}")
                    content = z.read(matches[0])
                    validate_ts(content, expected)
                    found[name] = content
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for name, content in found.items():
                (destination / name).write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            provenance = {
                "dataset": "UCR TwoPatterns", "source_url": url,
                "original_split": True, "length": 128, "classes": 4,
                "train_count": 1000, "test_count": 4000, "sha256": hashes,
                "note": "Official original sequences, NOT generated/simulated by this repository."
            }
            (destination / "SOURCE.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(provenance, indent=2))
            return
        except Exception as exc:
            last_error = exc
            print(f"Download attempt failed: {url}: {exc}")
    raise RuntimeError(f"All official TwoPatterns download URLs failed: {last_error}")


if __name__ == "__main__":
    download()
