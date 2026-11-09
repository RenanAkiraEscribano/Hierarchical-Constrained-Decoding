"""Dataset metadata, immutable ontologies and image loading for the five datasets."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import PurePosixPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, DATA_ROOT, DATASETS, ONTOLOGY_ROOT, ROOT, WORKERS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=5)
def dataset_spec(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    entries = json.loads((ONTOLOGY_ROOT / "datasets.json").read_text(encoding="utf-8"))
    return next(entry for entry in entries if entry["dataset"] == dataset)


def load_ontology(dataset):
    """Load the paper's exact tuple order and label IDs without rebuilding them."""
    spec = dataset_spec(dataset)
    path = ROOT / spec["valid_tuples_path"]
    if sha256(path) != spec["valid_tuples_sha256"]:
        raise ValueError(f"Ontology checksum mismatch: {dataset}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        levels = next(reader)
        tuples = [tuple(map(int, row)) for row in reader]
    if levels != spec["levels"] or tuples != sorted(set(tuples)):
        raise ValueError(f"Ontology order or columns changed: {dataset}")
    if len(tuples) != spec["tuple_count"]:
        raise ValueError(f"Unexpected tuple count: {dataset}")
    mapping_path = ROOT / spec["mapping_path"]
    if sha256(mapping_path) != spec["mapping_sha256"]:
        raise ValueError(f"Label mapping checksum mismatch: {dataset}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    counts = spec["class_counts"]
    for depth, key in enumerate(spec["mapping_levels"]):
        if any(not 0 <= row[depth] < counts[depth] or str(row[depth]) not in mapping[key]
               for row in tuples):
            raise ValueError(f"Invalid label ID: {dataset}/{key}")
    return levels, tuples, counts


def read_rows(dataset):
    path = DATA_ROOT / dataset / "metadata/dataset.csv"
    expected_levels, tuples, counts = load_ontology(dataset)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        levels = [name for name in fields if name not in ("split", "filename")]
        if levels != expected_levels or not {"split", "filename"}.issubset(fields):
            raise ValueError(f"Unexpected metadata columns: {path.name}")
        rows = list(reader)
    spec = dataset_spec(dataset)
    if Counter(row["split"] for row in rows) != spec["metadata_row_counts"]:
        raise ValueError(f"Split sizes differ from the paper: {dataset}")
    valid = set(tuples)
    for row in rows:
        labels = tuple(int(row[level]) for level in levels)
        if labels not in valid or any(not 0 <= value < count for value, count in zip(labels, counts)):
            raise ValueError(f"A label tuple is outside the paper ontology: {dataset}")
        filename = PurePosixPath(row["filename"].replace("\\", "/"))
        if filename.is_absolute() or ".." in filename.parts or not filename.parts:
            raise ValueError("Image filenames must be relative to the dataset images directory")
    return path, levels, rows


def validate_dataset(dataset):
    path, levels, rows = read_rows(dataset)
    _, tuples, counts = load_ontology(dataset)
    return {"dataset": dataset, "levels": levels, "classes": counts,
            "tuples": len(tuples), "split_sizes": dict(Counter(row["split"] for row in rows)),
            "metadata_matches_original": sha256(path) == dataset_spec(dataset)["metadata_sha256"]}


class ImageRows(Dataset):
    def __init__(self, dataset, rows, levels, tuples, transform):
        self.root = DATA_ROOT / dataset / "images"
        self.rows = rows
        self.levels = levels
        self.tuple_to_id = {value: index for index, value in enumerate(tuples)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labels = tuple(int(row[level]) for level in self.levels)
        parts = PurePosixPath(row["filename"].replace("\\", "/")).parts
        with Image.open(self.root.joinpath(*parts)) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor(labels, dtype=torch.long), self.tuple_to_id[labels]


def make_loader(dataset, rows, levels, tuples, transform, split, device="cuda", workers=WORKERS):
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"Empty split: {dataset}/{split}")
    return DataLoader(ImageRows(dataset, selected, levels, tuples, transform),
                      batch_size=BATCH_SIZE, shuffle=(split == "train"), num_workers=workers,
                      pin_memory=torch.device(device).type == "cuda", persistent_workers=False)
