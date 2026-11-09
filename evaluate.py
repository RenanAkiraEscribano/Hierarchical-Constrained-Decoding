"""Evaluate the paper's main decoders on newly supplied final checkpoints."""
import argparse
import json

from hcd.config import BACKBONES, DATASETS, METHODS, SEEDS
from hcd.evaluation import evaluate_checkpoint
from hcd.experiment import run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="CUB-200-2011")
    parser.add_argument("--backbone", choices=(*BACKBONES, "all"), default="dinov3")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="independent")
    parser.add_argument("--seed", choices=(*map(str, SEEDS), "all"), default="all")
    parser.add_argument("--checkpoint", help="Optional checkpoint path relative to the checkout")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Default: CUDA when available, otherwise CPU")
    args = parser.parse_args()
    if args.checkpoint:
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.device)), flush=True)
        return
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
    methods = METHODS if args.method == "all" else (args.method,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                for seed in seeds:
                    checkpoint = run_directory(dataset, backbone, method, "final", seed) / "best.pt"
                    print(json.dumps(evaluate_checkpoint(checkpoint, args.device)), flush=True)


if __name__ == "__main__":
    main()
