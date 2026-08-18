# PTB-XL ECG Classification

An end-to-end PyTorch project for multi-label classification of 12-lead electrocardiograms from PTB-XL. The pipeline maps SCP-ECG statements to five diagnostic superclasses, applies training-derived per-lead normalization, and compares a compact baseline CNN with a class-balanced residual 1D CNN.

## Results

The experiments use the official PTB-XL split: folds 1-8 for training, fold 9 for validation, and fold 10 as the held-out test set. The residual model was trained once with random seed `42`; repeated-seed confidence intervals remain future work.

| Model and decision rule | Macro F1 | Micro F1 | Macro AUROC | Micro AUROC |
|---|---:|---:|---:|---:|
| Calibrated baseline CNN | 0.729 | 0.761 | 0.916 | 0.927 |
| Residual CNN, fixed threshold 0.5 | **0.738** | **0.767** | **0.917** | **0.929** |
| Residual CNN, validation-selected thresholds | 0.734 | 0.766 | 0.917 | 0.929 |

Validation-selected thresholds improved the treatment of individual minority classes but did not improve aggregate test F1 over the fixed threshold. Both results are retained to show that threshold choices made on validation data do not always transfer perfectly to the test set.

## Dataset

This project uses [PTB-XL v1.0.3 from PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), a collection of 21,799 clinical ECG recordings from 18,869 patients. Each recording contains a 10-second, 12-lead ECG. The experiments use the 100 Hz signals with shape `(12, 1000)`.

The task covers five diagnostic superclasses:

- `NORM`: normal ECG
- `MI`: myocardial infarction
- `STTC`: ST/T changes
- `CD`: conduction disturbance
- `HYP`: hypertrophy

The dataset is not committed to this repository. After downloading and extracting PTB-XL, the expected metadata location is:

```text
data/ptb-xl/ptbxl_database.csv
data/ptb-xl/scp_statements.csv
data/ptb-xl/records100/
```

## Method

The reusable pipeline provides:

- Official folds 1-8/9/10 for training, validation, and test data
- Training-only, per-lead normalization statistics
- Lazy WFDB waveform loading through a PyTorch `Dataset`
- Multi-hot targets for the five diagnostic superclasses
- Softened inverse-frequency weights for imbalanced classes
- AdamW optimization, learning-rate reduction, checkpointing, and early stopping
- Fixed and validation-selected decision thresholds
- Per-class confusion counts, precision, sensitivity, specificity, F1, and AUROC

The residual model contains approximately 1.84 million trainable parameters. It uses six residual convolutional blocks, global average pooling, and dropout before the five-label classifier.

## Repository structure

```text
notebooks/
  01_exploration.ipynb         Dataset and annotation exploration
  02_data_pipeline.ipynb       Signal loading and normalization pipeline
  03_model_prototype.ipynb     Baseline CNN experiment
  04_improved_training.ipynb   Residual CNN and threshold experiment
src/
  data.py                      Metadata, splits, class weights, and loaders
  dataset.py                   Lazy-loading PyTorch dataset
  evaluation.py                Inference, thresholds, and metrics
  model.py                     Baseline and residual CNN architectures
  preprocessing.py             WFDB loading and normalization
  training.py                  Training, checkpointing, and early stopping
tests/                         Focused unit tests for the reusable pipeline
```

## Setup

The project was tested with Python `3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks are intended to be read in numerical order. Notebook 03 creates the baseline checkpoint used for the comparison in Notebook 04. Model checkpoints use the `.pt` extension and are intentionally excluded from version control; the saved notebook outputs preserve the reported experiment results.

Run the automated checks from the project root with:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

Notebook 04 records the experiment configuration, selected device, training history, best epoch, validation-selected thresholds, per-class test results, and aggregate baseline comparison. Random number generators are seeded through the reusable training module.

Exact dependency versions are recorded in `requirements.txt`. Hardware and backend differences can still introduce small numerical variation.

## Limitations

- Results come from one random seed and do not include uncertainty intervals.
- Evaluation is limited to the PTB-XL held-out fold; no external clinical dataset has been tested.
- Diagnostic superclasses are broad multi-label targets derived from dataset annotations.
- The software is an educational machine-learning project and is not a medical device or a substitute for professional clinical interpretation.

## Citation

Wagner, Patrick, et al. “PTB-XL, a large publicly available electrocardiography dataset” (version 1.0.3). PhysioNet (2022). [https://doi.org/10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)
