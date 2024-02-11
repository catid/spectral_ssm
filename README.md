# Spectral State Space Models

Work in progress implementation of "Spectral State Space Models" (Agarwal, 2024).

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

```bash
git clone https://github.com/catid/spectral_ssm
cd spectral_ssm

conda create -n sssm python=3.10 -y && conda activate sssm

pip install -U -r requirements.txt

# Pregenerate Hankel spectra
python hankel_spectra.py
```

