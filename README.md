# Hierarchical Constrained Decoding

Core implementation of **Hierarchical Constrained Decoding for Consistent
Multi-Level Fine-Grained Classification under Nested and Crossed Label
Structures**.

The repository contains the models, dataset ontologies, training procedure,
decoders, and evaluation functions for the paper's main experiment. Dataset
images, per-image metadata, pretrained weights, and trained checkpoints are
obtained separately. Reported results are not embedded in the code.

## Method

HCD chooses a complete valid label tuple using

```text
score(v | x) = z_K(v_K) / T_K + alpha * sum_{k < K} z_k(v_k) / T_k
prediction   = argmax over valid tuples v
```

Each temperature is fitted by marginal negative log-likelihood on validation
data, with `T_k` in `[0.05, 20]`. Alpha is then selected by validation
exact-tuple accuracy from `{0+, 1/8, 1/4, 1/2, 1, 2, 4}`. Ties favor the
candidate closest to 1 in log2 distance, then the smaller candidate.

`0+` is the lexicographic limit: retain the finest-level argmax and select its
compatible ancestors by their scaled score. It is represented by `0.0` in the
API. With unit temperatures, this endpoint is F2C and `alpha=1` is JO.
Already-valid independent predictions are preserved, including score ties.

The main comparison includes:

| Training configuration | Evaluated prediction rules |
| --- | --- |
| `independent` | Independent, C2F, F2C, JO, HCD |
| `joint` | One classification head over complete valid tuples |
| `regularized` | Independent regularized heads, Reg. + F2C, Reg. + JO, Reg. + HCD |

Regularized training uses mean per-level cross-entropy plus hierarchical KL
regularization with coefficient `0.1`. Parent-conditioned targets are uniform
over the children allowed by the ontology, including crossed structures.

## Models

All five frozen image towers are ViT-B/16 models at 224 px, with randomly
initialized linear heads. LoRA uses rank 16, scale 16, and no dropout on the
`qkv` and output projections of all 12 attention blocks: 884,736 trainable
adapter parameters per backbone. Text encoders are unused, and pretrained
attention pooling remains frozen.

| CLI identifier | `timm` model |
| --- | --- |
| `augreg` | `vit_base_patch16_224.augreg_in21k` |
| `dinov3` | `vit_base_patch16_dinov3.lvd1689m` |
| `clip` | `vit_base_patch16_clip_224.openai` |
| `siglip2` | `vit_base_patch16_siglip_224.v2_webli` |
| `pe_core` | `vit_pe_core_base_patch16_224.fb` |

## Datasets

| Dataset identifier | Levels, coarse to fine | Classes | Valid tuples |
| --- | --- | --- | ---: |
| `CUB-200-2011` | order / family / genus / scientific_name | 13 / 38 / 127 / 200 | 200 |
| `EU-Moths` | family / genus / species | 11 / 143 / 200 | 200 |
| `FGVC-Aircraft` | manufacturer / family / variant | 30 / 70 / 100 | 100 |
| `Stanford-Cars-196` | type / make / model | 9 / 49 / 196 | 196 |
| `UFPR-VeSV` | type / make / model | 14 / 26 / 136 | 225 |

Ontology CSVs and ID-to-name mappings are in [hierarchies/](hierarchies/).
Their identifiers, numeric lexicographic tuple order, and original bytes are
preserved. [datasets.json](hierarchies/datasets.json) records the split sizes
and the ontology, mapping, and original metadata hashes. Stanford Cars uses
model-year fine classes; its mapping key is `model+year`.

**UFPR-VeSV assumes a known label space:** its ontology contains the 225 tuples
observed across training, validation, and test, including eight absent from
training. Evaluation images are not used by the optimizer; the structure of
their labels is included in the ontology. The other four ontologies contain
training-observed tuples and cover all evaluation labels.

Obtain images from the original providers:
[CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/),
[EU-Moths](https://github.com/cvjena/eu-moths-dataset),
[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/),
[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), and
[UFPR-VeSV](https://github.com/Lima001/UFPR-VeSV-Dataset).
Dataset citations are provided in [references/datasets.bib](references/datasets.bib).

Install each dataset using this layout:

```text
datasets/<dataset-identifier>/
├── images/
└── metadata/
    └── dataset.csv
```

Each per-image CSV must contain `split`, `filename`, and the level columns in
the order above. Splits are `train`, `val`, and `test`; image filenames are
relative to `images/`; labels are integer IDs from the supplied mappings.
Preserve the original labels, splits, and filenames. Exact replication
requires the per-image metadata matching the hashes in `datasets.json`;
these per-image CSVs are not distributed here.

## Installation

Use Python 3.12 and the packages in [requirements.txt](requirements.txt).
For a Linux shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The package versions correspond to the experimental environment, including
PyTorch 2.6.0 with CUDA 12.4 and `timm` 1.0.28. Training uses CUDA and mixed
precision. The decoder and metric functions operate on NumPy arrays on CPU.
Paths are resolved from the checkout automatically; no private filesystem or
machine identifiers are required.

## Main experiment

Validate all five datasets and their ontologies without training:

```bash
python train.py --dataset all --prepare
```

Train one dataset/backbone/configuration:

```bash
python train.py --dataset CUB-200-2011 --backbone dinov3 --method independent
```

Each configuration searches learning rates `1e-4`, `3e-4`, and `1e-3` with seed
11, selects by validation exact-tuple accuracy, then trains final seeds
11, 23, and 37. Tied learning rates favor the smaller value. The training
implementation uses AdamW, weight decay `0.01`, batch size 8, head learning
rate `10 ×` the LoRA rate, cosine scheduling, gradient clipping at norm 1,
and at most 30 epochs. The best validation checkpoint is retained.

Training augmentation is a random resized crop with scale `[0.8, 1]` and
horizontal flip. Evaluation uses resize 256, center crop 224, bicubic
interpolation, and the backbone's pretrained normalization.

Evaluate the three final seeds of the same configuration:

```bash
python evaluate.py --dataset CUB-200-2011 --backbone dinov3 --method independent
```

`--dataset all --backbone all --method all` runs the complete main comparison.
Individual evaluation also accepts `--seed 11` or `--checkpoint` with a
checkout-relative checkpoint path. Evaluation chooses CUDA when available;
`--device cpu` explicitly selects CPU inference.

Evaluation fits HCD on validation and saves its frozen parameters before
reading test images. It computes exact-tuple accuracy, per-level accuracy,
invalid-tuple rate, and recovery of initially invalid predictions. Rates are
fractions in the returned JSON. It does not generate manuscript tables or
precomputed result summaries. The validation set is shared with learning-rate
and checkpoint selection, as in the paper; test labels never select parameters.

Checkpoints, training records, and fitted decoder parameters are created under
`runs/`, which is excluded from version control. Independent configurations
should be scheduled sequentially on the selected GPU.

## Core API

```python
from hcd.decoding import fit_hcd, hcd

# validation_logits/test_logits: one [N, classes_at_level] array per level
# validation_labels: [N, levels]; valid_tuples: [valid_tuple_count, levels]
parameters = fit_hcd(validation_logits, validation_labels, valid_tuples)
predictions = hcd(test_logits, valid_tuples, parameters)
```

## Structure

```text
train.py              # Learning-rate search and three-seed training
evaluate.py           # Main checkpoint/decoder comparison
hcd/
  config.py           # Five datasets, five backbones, and fixed hyperparameters
  data.py             # Metadata, ontologies, and image loading
  models.py           # Frozen backbones, LoRA, heads, and transforms
  training.py         # Losses, regularization, optimizer, and epoch operations
  experiment.py       # Training/checkpoint workflow
  decoding.py         # Independent, C2F, F2C, JO, joint head, and HCD
  metrics.py          # Per-input accuracy, validity, and recovery
  evaluation.py       # Validation fitting followed by test inference
hierarchies/          # Five ontologies and label-name mappings
references/           # Dataset bibliography
```

Verify the published files from the checkout root with
`sha256sum -c SHA256SUMS`. The main implementation contains no dataset images,
campaign protocols, historical rerun scripts, manuscript result tables, or
additional calibration experiments.
