# CHIMERA

## Environment Setup

Create and activate the conda environment from the provided configuration file:

```bash
conda env create -f environment.yml
conda activate chimera
```

---

## Running Tasks

### Single GPU

```bash
python scripts/run_task.py --config configs/dataB_regression.yaml
```

### Multi-GPU (4 GPUs example)

```bash
torchrun --standalone --nproc_per_node=4 scripts/run_task.py --config configs/dataB_regression.yaml
```

---

## Available Tasks & Configurations

| Task Type      | Dataset A                          | Dataset B                          |
|----------------|------------------------------------|------------------------------------|
| Regression     | `configs/dataA_regression.yaml`    | `configs/dataB_regression.yaml`    |
| Classification | `configs/dataA_classification.yaml`| `configs/dataB_classification.yaml`|
