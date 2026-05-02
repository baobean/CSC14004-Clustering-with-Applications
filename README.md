# CSC14004 — Federated Clustering with SR-FCA

Course project reproducing **SR-FCA** (Vardhan, Ghosh, Mazumdar, TMLR 02/2024) — a bottom-up clustered federated learning algorithm that discovers the number of clusters `K` from a pairwise-distance graph instead of requiring it as input.

## Environment requirements

- Python ≥ 3.10 (3.11 recommended).
- CPU only — no GPU required. The four notebooks run end-to-end in **~13 hours** of wall clock on a single laptop CPU.

## Installation

Either conda or venv works.

```bash
# Option A: conda
conda create -n srfca python=3.11 && conda activate srfca
pip install -r requirements.txt

# Option B: venv
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

MNIST and Fashion-MNIST are downloaded automatically by `torchvision.datasets` into `./data/` on first use; no manual data setup required.

## Folder structure
```
Group_01/
├── README.md
├── src/
├── notebooks/
│   ├── 01_main_experiments.ipynb
│   ├── 02_ablation_study.ipynb
│   ├── 03_new_dataset.ipynb
│   └── 04_application.ipynb
├── data/
├── results/
├── paper/
│   └── paper.pdf
└── docs/
    └── report.pdf
```

## Running instructions

```bash
# 1. Sanity check (~30 s; should print misclustering_error = 0)
python -m src.srfca --smoke-test --seed 0

# 2. Run the four notebooks (any order; resumable on interrupt)
```

Each notebook reads/writes `results/main.csv`, `results/ablation_A1.csv`, `results/fashion_mnist.csv`, `results/part_c.csv` and the corresponding PNGs under `results/figures/`.
