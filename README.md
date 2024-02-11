# Spectral State Space Models (Work in Progress Do not use)

Implementation of "Spectral State Space Models" (Agarwal, 2024): https://arxiv.org/pdf/2312.06837.pdf which was based on "Learning Linear Dynamical Systems via Spectral Filtering" (Hazan, 2017) https://arxiv.org/pdf/1711.00946.pdf

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

```bash
git clone https://github.com/catid/spectral_ssm
cd spectral_ssm

conda create -n sssm python=3.10 -y && conda activate sssm

pip install -U -r requirements.txt

# Pregenerate Hankel spectra
python hankel_spectra_test.py

# Test convolutions
python convolutions_test.py

# Test AR-STU layer
python ar_stu_test.py
```
