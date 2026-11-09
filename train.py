"""Train the paper's three configurations using validation-selected learning rates."""
import argparse
import json

from hcd.config import BACKBONES, DATASETS, METHODS
from hcd.data import validate_dataset
from hcd.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="CUB-200-2011")
    parser.add_argument("--backbone", choices=(*BACKBONES, "all"), default="dinov3")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="independent")
    parser.add_argument("--prepare", action="store_true", help="Validate metadata and ontologies without training")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if args.prepare:
        for dataset in datasets:
            print(json.dumps(validate_dataset(dataset)), flush=True)
        return
    backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
    methods = METHODS if args.method == "all" else (args.method,)
    for dataset in datasets:
        for backbone in backbones:
            for method in methods:
                run_experiment(dataset, backbone, method)


if __name__ == "__main__":
    main()
