"""Fixed configuration of the paper's main experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"
ONTOLOGY_ROOT = ROOT / "hierarchies"
RUN_ROOT = ROOT / "runs"

DATASETS = ("CUB-200-2011", "EU-Moths", "FGVC-Aircraft", "Stanford-Cars-196", "UFPR-VeSV")
BACKBONES = {
    "augreg": "vit_base_patch16_224.augreg_in21k",
    "dinov3": "vit_base_patch16_dinov3.lvd1689m",
    "clip": "vit_base_patch16_clip_224.openai",
    "siglip2": "vit_base_patch16_siglip_224.v2_webli",
    "pe_core": "vit_pe_core_base_patch16_224.fb",
}
METHODS = ("independent", "joint", "regularized")
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
SEEDS = (11, 23, 37)
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 8
WORKERS = 4
RANK = 16
LORA_ALPHA = 16
LAMBDA_KL = 0.1
WEIGHT_DECAY = 0.01
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
