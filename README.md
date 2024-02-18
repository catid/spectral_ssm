# Spectral State Space Models

Implementation of "Spectral State Space Models" (Agarwal, 2024): https://arxiv.org/pdf/2312.06837.pdf which was based on "Learning Linear Dynamical Systems via Spectral Filtering" (Hazan, 2017) https://arxiv.org/pdf/1711.00946.pdf

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

```bash
git clone https://github.com/catid/spectral_ssm
cd spectral_ssm

conda create -n sssm python=3.10 -y && conda activate sssm

pip install -U -r requirements.txt

# Pre-generate Hankel spectra and run unit tests
pytest

# Download audio dataset for training
python download_dataset.py
```

## Train

A basic training script is included for audio sequence data.  This breaks up a FLAC audio file into segments with MFCC embedding as features, which is a tensor [B, L, D].  B=number of segments, L=length of segment in audio samples, D=12 MFCC embeddings.  The batches are split into test/train sets.

```bash
python train.py
```
