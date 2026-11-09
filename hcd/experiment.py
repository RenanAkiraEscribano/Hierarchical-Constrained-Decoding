"""Learning-rate selection and three-seed training for the main experiment."""
from __future__ import annotations

import gc
import json
import time

import torch

from .config import (BACKBONES, EPOCHS, LEARNING_RATES, METHODS, PATIENCE,
                     ROOT, RUN_ROOT, SEEDS)
from .data import dataset_spec, load_ontology, make_loader, read_rows, sha256
from .models import MultiHeadModel, transforms_for
from .training import kl_targets, make_optimizer, one_epoch, seed_all


def run_directory(dataset, backbone, method, stage, seed, lr=None):
    if backbone not in BACKBONES or method not in METHODS or stage not in ("search", "final"):
        raise ValueError("Unknown backbone, training method, or stage")
    suffix = f"search_lr{lr:g}_seed{seed}" if stage == "search" else f"final_seed{seed}"
    return RUN_ROOT / dataset / backbone / method / suffix


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_one(dataset, backbone, method, lr, seed, stage="final", max_epochs=EPOCHS, patience=PATIENCE):
    if not torch.cuda.is_available():
        raise RuntimeError("The paper's mixed-precision training requires CUDA")
    metadata_path, levels, rows = read_rows(dataset)
    _, tuples, counts = load_ontology(dataset)
    directory = run_directory(dataset, backbone, method, stage, seed, lr)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best.pt"
    complete = directory / "training.json"
    configuration = {
        "dataset": dataset, "backbone": backbone, "model_name": BACKBONES[backbone],
        "method": method, "lr": lr, "seed": seed, "stage": stage,
        "max_epochs": max_epochs, "patience": patience, "levels": levels,
        "class_counts": counts, "tuple_count": len(tuples),
        "metadata_sha256": sha256(metadata_path),
        "ontology_sha256": dataset_spec(dataset)["valid_tuples_sha256"],
    }
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved["configuration"] != configuration or sha256(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError("Existing run has different data, configuration, or checkpoint")
        return saved

    seed_all(seed)
    model = MultiHeadModel(BACKBONES[backbone], counts, len(tuples), method).cuda()
    train_transform, valid_transform, _ = transforms_for(model)
    train_loader = make_loader(dataset, rows, levels, tuples, train_transform, "train")
    valid_loader = make_loader(dataset, rows, levels, tuples, valid_transform, "val")
    targets = kl_targets(tuples, counts, "cuda")
    optimizer = make_optimizer(model, lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best, best_epoch, stale = -1.0, 0, 0
    started = time.time()
    epochs_path = directory / "epochs.jsonl"
    epochs_path.write_text("", encoding="utf-8")
    try:
        for epoch in range(1, max_epochs + 1):
            train = one_epoch(model, train_loader, method, targets, optimizer, scaler)
            validation = one_epoch(model, valid_loader, method, targets)
            scheduler.step()
            event = {"epoch": epoch, "train": train, "validation": validation}
            with epochs_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event) + "\n")
            print(json.dumps({"run": directory.relative_to(ROOT).as_posix(), **event}), flush=True)
            if validation["exact"] > best:
                best, best_epoch, stale = validation["exact"], epoch, 0
                state = {name: value.cpu() for name, value in model.state_dict().items()
                         if name.startswith("heads.") or ".down." in name or ".up." in name}
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"model": state, "configuration": configuration,
                            "epoch": epoch, "validation_exact": best}, temporary)
                temporary.replace(checkpoint)
            else:
                stale += 1
            if patience is not None and stale >= patience:
                break
        summary = {"configuration": configuration, "best_epoch": best_epoch,
                   "validation_exact": best, "epochs_completed": epoch,
                   "elapsed_seconds": time.time() - started,
                   "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                   "checkpoint_sha256": sha256(checkpoint)}
        write_json(complete, summary)
        return summary
    finally:
        del model, optimizer, scheduler, scaler, targets, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()


def run_experiment(dataset, backbone, method):
    """Search on seed 11, select by validation accuracy, then train seeds 11/23/37."""
    candidates = []
    for lr in LEARNING_RATES:
        directory = run_directory(dataset, backbone, method, "search", SEEDS[0], lr)
        failed = directory / "failure.json"
        if failed.exists():
            continue
        try:
            candidates.append(train_one(dataset, backbone, method, lr, SEEDS[0], "search"))
        except FloatingPointError as error:
            write_json(failed, {"dataset": dataset, "backbone": backbone, "method": method,
                                "lr": lr, "seed": SEEDS[0], "reason": str(error)})
    if not candidates:
        raise RuntimeError("All learning-rate searches produced nonfinite losses")
    selected = max(candidates, key=lambda item: (item["validation_exact"], -item["configuration"]["lr"]))
    lr = selected["configuration"]["lr"]
    print(json.dumps({"selected_lr": lr, "dataset": dataset,
                      "backbone": backbone, "method": method}), flush=True)
    for seed in SEEDS:
        train_one(dataset, backbone, method, lr, seed, "final")
