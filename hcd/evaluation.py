"""Validation-fitted HCD and per-checkpoint evaluation of the main comparison."""
from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import BACKBONES, ROOT
from .data import dataset_spec, load_ontology, make_loader, read_rows, sha256
from .decoding import HCDParameters, c2f, f2c, fit_hcd, hcd, independent, jo, joint_head
from .experiment import write_json
from .metrics import prediction_metrics
from .models import MultiHeadModel, transforms_for
from .training import seed_all


def collect_logits(model, loader, device):
    device = torch.device(device)
    pieces, labels_out = None, []
    model.eval()
    with torch.inference_mode():
        for images, labels, _ in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                heads = model(images.to(device, non_blocking=True))
            if pieces is None:
                pieces = [[] for _ in heads]
            for level, head in enumerate(heads):
                if not torch.isfinite(head).all():
                    raise FloatingPointError("Nonfinite inference logits")
                pieces[level].append(head.float().cpu().numpy())
            labels_out.append(labels.numpy())
    if pieces is None:
        raise ValueError("Cannot evaluate an empty split")
    return [np.concatenate(part) for part in pieces], np.concatenate(labels_out)


def evaluate_checkpoint(checkpoint_path, device=None):
    path = Path(checkpoint_path)
    path = path if path.is_absolute() else ROOT / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    if configuration["stage"] != "final":
        raise ValueError("Test evaluation requires a final-seed checkpoint")
    if configuration["model_name"] != BACKBONES.get(configuration["backbone"]):
        raise ValueError("Checkpoint backbone is outside the paper's model configuration")
    dataset, method = configuration["dataset"], configuration["method"]
    metadata_path, levels, rows = read_rows(dataset)
    _, tuples, counts = load_ontology(dataset)
    if (sha256(metadata_path) != configuration["metadata_sha256"] or
            dataset_spec(dataset)["valid_tuples_sha256"] != configuration["ontology_sha256"] or
            levels != configuration["levels"] or counts != configuration["class_counts"] or
            len(tuples) != configuration["tuple_count"]):
        raise ValueError("Checkpoint and dataset metadata/ontology differ")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(configuration["seed"])
    model = MultiHeadModel(configuration["model_name"], counts, len(tuples), method)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(checkpoint["model"]) != expected:
        raise ValueError("Checkpoint must contain exactly the LoRA and classification-head parameters")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    _, transform, _ = transforms_for(model)
    try:
        parameters = None
        if method != "joint":
            calibration_path = path.parent / "hcd_parameters.json"
            identity = {"checkpoint_sha256": sha256(path),
                        "metadata_sha256": configuration["metadata_sha256"],
                        "ontology_sha256": configuration["ontology_sha256"],
                        "decoder_sha256": sha256(Path(__file__).with_name("decoding.py"))}
            if calibration_path.exists():
                saved = json.loads(calibration_path.read_text(encoding="utf-8"))
                if saved["identity"] != identity:
                    raise ValueError("Saved HCD parameters belong to different inputs")
                parameters = HCDParameters(tuple(saved["parameters"]["temperatures"]),
                                           saved["parameters"]["alpha"])
            else:
                validation_loader = make_loader(dataset, rows, levels, tuples, transform, "val", device)
                validation_logits, validation_labels = collect_logits(model, validation_loader, device)
                parameters = fit_hcd(validation_logits, validation_labels, tuples)
                # Persist and freeze validation-fitted parameters before reading test images.
                write_json(calibration_path, {"identity": identity, "parameters": asdict(parameters)})
                del validation_loader, validation_logits, validation_labels

        test_loader = make_loader(dataset, rows, levels, tuples, transform, "test", device)
        logits, labels = collect_logits(model, test_loader, device)
        if method == "joint":
            measurements = {"joint_head": prediction_metrics(joint_head(logits[0], tuples), labels, tuples)}
        else:
            base = independent(logits)
            prefix = "regularized" if method == "regularized" else "independent"
            measurements = {prefix: prediction_metrics(base, labels, tuples)}
            decoders = {"f2c": f2c(logits, tuples), "jo": jo(logits, tuples),
                        "hcd": hcd(logits, tuples, parameters)}
            if method == "independent":
                decoders = {"c2f": c2f(logits, tuples), **decoders}
            for name, predicted in decoders.items():
                key = "regularized+" + name if method == "regularized" else name
                measurements[key] = prediction_metrics(predicted, labels, tuples, reference=base)
        return {"dataset": dataset, "backbone": configuration["backbone"],
                "training_method": method, "seed": configuration["seed"],
                "hcd_parameters": asdict(parameters) if parameters is not None else None,
                "measurements": measurements}
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
