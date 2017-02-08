# CZ4041
Course project private repository

## Environment
* Miniconda (or Anaconda) **64-bit version** [link](https://conda.io/miniconda.html)
  * After installation, create 2 conda environment:
    1. Python 2.7
    2. Python 3.5
* Install `numpy`, `matplotlib`, and `pandas` to both environments
* (Optional) Install CUDA 8.0 with cuDNN v5.1 [link](https://developer.nvidia.com/cuda-toolkit)
* **(Important)** Install `jupyter` with `conda install jupyter`
* **(Important)** Install `scikit-learn` with `conda install scikit-learn`

## First step
1. Download and extract from Kaggle `train.csv`, `test.csv`, and `store.csv` to data/ directory.
2. **(Important)** Extract `merged.tar.gz` to data/

## Note
* **(Important)** `src/vis.py` and `src/merge.py` are outdated/unfinished, use `src/vis.ipynb`, `src/merge.ipynb` and `src/prep.ipynb` instead!
 * `src/vis.ipynb`: for viz
 * `src/merge.ipynb`: for cleaning and merging `train.csv` with `store.csv`
 * `src/prep.ipynb`: for cleaning and merging `test.csv` with `store.csv`
