# BERT vs. RoBERTa: Sentiment Classification on Twitter Data

## Overview

This project fine-tunes and compares two pretrained Transformer encoders,
`bert-base-uncased` and `roberta-base`, on a four-class Twitter sentiment
classification task. The notebooks use the same training loop, hyperparameters,
label mapping, and validation split for both models so the comparison is
directly comparable.

The repository contains the original executed notebooks and the metrics
recovered from those runs. No new training run or synthetic result has been
added.

## Problem statement

Given a tweet and its topic/entity, predict one of:

- `Irrelevant`
- `Negative`
- `Neutral`
- `Positive`

This is a supervised multi-class text classification problem.

## Why compare BERT and RoBERTa?

BERT and RoBERTa share the Transformer encoder architecture, but RoBERTa was
pretrained with a revised training recipe and corpus. Holding the downstream
training setup constant provides a practical comparison of their performance
on the same sentiment task.

## Dataset

The notebooks expect the Kaggle **Twitter Entity Sentiment Analysis** CSV files:

<https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis>

Download `twitter_training.csv` and `twitter_validation.csv` and place them in
`data/`. The files are read as headerless columns in this order:
`id, topic, label, text`.

The executed runs report 1,000 validation examples and four labels. The
dataset is not included in this repository.

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- pandas and NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Methodology

1. Load the headerless training and validation CSV files.
2. Build the label mapping from the training labels and apply it to validation
   labels.
3. Tokenize text with each model's pretrained fast tokenizer.
4. Pad/truncate sequences to 128 tokens.
5. Fine-tune a sequence-classification head for three epochs with batch size
   16, AdamW, and learning rate `2e-5`.
6. Evaluate after each epoch on the validation split.
7. Report accuracy, per-class precision/recall/F1, the confusion matrix, and
   learning curves.

The two notebooks use the same basic pipeline; only the pretrained checkpoint
and tokenizer differ.

## Results

These values are taken from the stored notebook outputs and represent one
executed run on the validation split:

| Model | Accuracy | Macro F1 | Weighted F1 | Best validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| BERT (`bert-base-uncased`) | 0.960 | 0.96 | 0.96 | 0.960 |
| RoBERTa (`roberta-base`) | 0.971 | 0.97 | 0.97 | 0.971 |

RoBERTa was 1.1 percentage points higher in validation accuracy in the saved
metrics table. Its final validation loss was also lower (`0.1060` vs.
`0.1643`). These are single-run validation results, not a statistically
repeated benchmark.

Detailed epoch histories are available in
[`results/bert_metrics.csv`](results/bert_metrics.csv) and
[`results/roberta_metrics.csv`](results/roberta_metrics.csv). The class-level
reports and confusion matrices remain in the executed notebooks.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the CUDA version
installed on your machine from <https://pytorch.org/get-started/locally/>.
The original recorded run used Python 3.10.11 and PyTorch 2.6.0 with CUDA 12.4
on an NVIDIA RTX 3060; CPU execution is also supported, but slower.

## How to run

1. Put the two dataset files in `data/` as described above.
2. Open either notebook in Jupyter or VS Code.
3. Run the cells from top to bottom:

   - [`Bert Model.ipynb`](Bert%20Model.ipynb)
   - [`RoBERTa Model.ipynb`](RoBERTa%20Model.ipynb)

The notebooks download pretrained checkpoints from Hugging Face on first use.
They save fine-tuned checkpoints to `bert_finetuned/` or
`roberta_finetuned/` when executed. Those generated directories are ignored by
Git.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── Bert Model.ipynb
├── RoBERTa Model.ipynb
├── data/                  # local dataset files; not committed
├── results/               # metrics recovered from executed notebooks
├── Link of the dataset.pdf
└── Comparative Analysis of Sentiment Analysis on Twitter Data between BERT and RoBERTa.pdf
```

## Limitations

- The comparison is based on one recorded run and one validation split; no
  confidence intervals or multiple random seeds were reported.
- The BERT notebook contains two recorded training traces with small
  differences; the surfaced CSV and comparison table use the later saved
  metrics table (`0.960` accuracy, `0.164303` final validation loss).
- The notebooks do not include the original CSV files, so a fresh run requires
  downloading the dataset.
- The validation split is used for model selection/monitoring, so it should not
  be described as an untouched test benchmark.
- The original run used a fixed PyTorch seed, but did not document every
  possible CUDA/dataloader determinism setting.
- Results may vary with hardware, library versions, and checkpoint downloads.

## Future improvements

- Add a separate untouched test split and repeated-seed confidence intervals.
- Track experiment configuration and runtime in a machine-readable results
  file.
- Add macro-F1-driven checkpoint selection and early stopping.
- Report parameter counts, training time, and peak memory for both models.
- Add automated smoke tests for data loading and metric generation.