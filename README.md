# DIGER: Differentiable Semantic ID for Generative Recommendation

This work has been **accepted as a full paper** at **SIGIR 2026**.

![SID](https://img.shields.io/badge/Task-SID-red)
![Generative recommendation](https://img.shields.io/badge/Task-Generative--recommendation-red)
<a href="https://arxiv.org/abs/2601.19711" alt="arXiv"><img src="https://img.shields.io/badge/arXiv-2601.19711-FAA41F.svg?style=flat" /></a>
<a href="https://mp.weixin.qq.com/s/Cs2kwRR0U94GyT5h7hkldg" alt="Chinese blog"><img src="https://img.shields.io/badge/blog-Xinzhiyuan-orange.svg?style=flat" /></a>

## Overview

DIGER studies differentiable semantic IDs for generative recommendation. This release contains the code, processed data, semantic embeddings, and RQ-VAE checkpoints needed to reproduce the paper rows for:

- **FrqUD**: frequency-based uncertainty decay.
- **SDUD**: standard-deviation uncertainty decay.
- **SDUD+FrqUD**: the combined setting.

The released scripts cover all three datasets used in the table: Beauty, Instruments, and Yelp.

## Repository Structure

```text
DIGER/
├── main.py
├── model.py
├── trainer.py
├── vq.py
├── data.py
├── config/
│   ├── beauty_jo.yaml
│   ├── instruments_jo.yaml
│   └── yelp_jo.yaml
├── dataset/
│   ├── beauty/
│   ├── instruments/
│   └── yelp/
├── rqvae_ckpt/
│   ├── beauty/best_collision_model.pth
│   ├── instruments/best_collision_model.pth
│   └── yelp/best_collision_model.pth
├── scripts/
│   ├── check_artifacts.py
│   ├── run_experiment.sh
│   ├── run_table_two_gpus.sh
│   └── verify_results.py
├── run_FrqUD.sh
├── run_SDUD.sh
├── run_SDUD_FrqUD.sh
└── run_reproduce_table.sh
```

Large release artifacts, including checkpoints, embeddings, and JSONL splits, are tracked with Git LFS.

## Requirements

```bash
conda create -n diger python=3.12.11 -y
conda activate diger
pip install -r requirements.txt
```

Reference environment used for the released paper logs:

- Python 3.12.11
- PyTorch 2.5.1
- Transformers 4.57.1
- Accelerate 1.10.1
- NumPy 2.3.1

Using newer major versions can change initialization and dropout RNG streams. A quick sanity check for the paper environment is the first Yelp SDUD+FrqUD training line:

```text
[Simple Uncertainty] sigma=2.0000, Loss=5.4814
```

If this line is closer to `Loss=5.5120`, the code and artifacts are likely correct but the active Python environment is not the paper environment.

After cloning the repository, pull the LFS files:

```bash
git lfs install
git lfs pull
```

Then verify the released artifacts:

```bash
python scripts/check_artifacts.py
```

## Data

Each dataset directory contains JSONL interaction splits, an item-id map, and the semantic embedding matrix used by the released configs:

```text
dataset/<dataset>/
├── <dataset>.train.jsonl
├── <dataset>.valid.jsonl
├── <dataset>.test.jsonl
├── <dataset>.emb_map.json
└── <Dataset>.emb-llama.npy
```

The loader expects each JSONL row to contain `inter_history` and `target_id`.

## Reproduction

Run one experiment:

```bash
bash scripts/run_experiment.sh beauty frqud
bash scripts/run_experiment.sh instruments sdud
bash scripts/run_experiment.sh yelp both
```

If you keep the paper environment outside your current shell, point the script at its `bin` directory:

```bash
DIGER_ENV_BIN=/path/to/env/bin bash scripts/run_experiment.sh yelp both
```

The same variable works for the full-table launcher:

```bash
DIGER_ENV_BIN=/path/to/env/bin bash run_reproduce_table.sh
```

Convenience wrappers default to Beauty and accept the dataset as the first argument:

```bash
bash run_FrqUD.sh beauty
bash run_SDUD.sh instruments
bash run_SDUD_FrqUD.sh yelp
```

Run the full paper table on at most two single-GPU processes. The worker script uses a shared task queue, so whichever GPU finishes first takes the next experiment:

```bash
bash run_reproduce_table.sh
```

By default this uses GPU `0` and GPU `1`. To choose another pair:

```bash
GPU_LIST="2 3" bash run_reproduce_table.sh
```

By default, `run_reproduce_table.sh` starts a fresh queue by resetting `reproduction_logs/table_queue.state`. To resume an interrupted queue, set:

```bash
RESUME_QUEUE=1 bash run_reproduce_table.sh
```

Training logs are written to `logs/<dataset>/`; stdout mirrors are written to `reproduction_logs/`. Model checkpoints are written to `myckpt/<dataset>/`.

## Paper Data

The paper reports these metrics (R@5/R@10/N@5/N@10):

- Beauty: FrqUD 0.0440/0.0683/0.0294/0.0372; SDUD 0.0442/0.0657/0.0292/0.0361; SDUD+FrqUD 0.0439/0.0696/0.0293/0.0376
- Instruments: FrqUD 0.0915/0.1138/0.0772/0.0844; SDUD 0.0905/0.1124/0.0753/0.0823; SDUD+FrqUD 0.0907/0.1127/0.0758/0.0829
- Yelp: FrqUD 0.0266/0.0432/0.0173/0.0227; SDUD 0.0267/0.0439/0.0171/0.0227; SDUD+FrqUD 0.0273/0.0437/0.0175/0.0227

After training, compare the newest matching logs with the paper targets:

```bash
python scripts/verify_results.py
```

The verifier scans `logs/*/*.log` and `reproduction_logs/*` (including `.driver.log`) from the current run. It uses a 1% relative tolerance with a small absolute floor for very small metrics.

If you only want to check the packaged reference logs without rerunning experiments, run:

```bash
VERIFY_WITH_BACKUP=1 python scripts/verify_results.py
```

## Citation

```bibtex
@article{fu2026differentiable,
  title={Differentiable Semantic ID for Generative Recommendation},
  author={Fu, Junchen and Ge, Xuri and Karatzoglou, Alexandros and Arapakis, Ioannis and Verberne, Suzan and Jose, Joemon M and Ren, Zhaochun},
  journal={arXiv preprint arXiv:2601.19711},
  year={2026}
}
```
