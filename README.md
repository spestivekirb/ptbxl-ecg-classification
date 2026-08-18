
## Dataset

**Dataset:** PTB-XL v1.0.3 from PhysioNet.

This project uses the PTB-XL v1.0.3 dataset, a large publicly available collection of 12-lead electrocardiograms distributed through PhysioNet.

PTB-XL contains 21,799 clinical ECG recordings from 18,869 patients, with diagnostic annotations mapped to standardized SCP-ECG statements.

For this project, the 100 Hz recordings are used and the task is defined over the five diagnostic superclasses: `NORM`, `MI`, `STTC`, `CD`, and `HYP`.

The dataset itself is not included in this repository. This repository contains only the code and analysis developed for this project.

## Current results

The baseline 1D CNN uses the official PTB-XL split: folds 1-8 for training, fold 9 for validation, and fold 10 as the held-out test set. Selecting per-class decision thresholds on the validation fold produced the following held-out test metrics:

- Macro F1: `0.729`
- Micro F1: `0.761`
- Macro AUROC: `0.916`
- Micro AUROC: `0.927`

The threshold calibration also raised `HYP` sensitivity to `0.691`. These results were measured from the saved baseline checkpoint. Notebook 04 defines the next residual-CNN experiment, whose results become available after execution.

## Project structure

- `src/data.py`: PTB-XL metadata preparation, official splits, class weights, datasets, and DataLoaders
- `src/preprocessing.py`: WFDB signal loading and training-only normalization statistics
- `src/model.py`: baseline and residual 1D CNN architectures
- `src/training.py`: device selection, reproducibility, epoch loops, checkpointing, scheduling, and early stopping
- `src/evaluation.py`: inference, validation-based threshold selection, and multi-label metrics
- `notebooks/03_model_prototype.ipynb`: completed baseline experiment
- `notebooks/04_improved_training.ipynb`: class-balanced residual-CNN experiment
- `tests/`: focused unit tests for the reusable pipeline

Run the automated checks from the project root with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Citation
Wagner, Patrick, et al. "PTB-XL, a large publicly available electrocardiography dataset" (version 1.0.3). PhysioNet (2022). RRID:SCR_007345. https://doi.org/10.13026/kfzx-aw45
