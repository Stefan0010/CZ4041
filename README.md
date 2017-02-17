# CZ4041
Course project private repository

## Environment
* Miniconda (or Anaconda) **64-bit version** [link](https://conda.io/miniconda.html)
  * After installation, create 2 conda environment:
    1. Python 2.7
    2. Python 3.5
* Install `numpy`, `matplotlib`, and `pandas` to both environments
* (Optional) Install CUDA 8.0 with cuDNN v5.1 [link](https://developer.nvidia.com/cuda-toolkit)
* Install `jupyter` with `conda install jupyter`
* Install `scikit-learn` with `conda install scikit-learn`

## First step
1. Download and extract from Kaggle `train.csv`, `test.csv`, and `store.csv` to _data/_ directory.
2. Extract `merged.tar.gz` to _data/_
3. Extract `splitted.tar.gz` to _data/_

## Note
* `src/vis.py` and `src/merge.py` are outdated/unfinished work, use `src/vis.ipynb`, `src/merge.ipynb` and `src/prep.ipynb` instead!
 * `src/vis.ipynb`: for viz
 * `src/merge.ipynb`: for cleaning and merging `train.csv` with `store.csv`
 * `src/prep.ipynb`: for cleaning and merging `test.csv` with `store.csv`
* Use 1000 stores (`data/1000_split.csv`) for training and 115 others for validation (`data/115_split.csv`)
* Use `util.py` to load dataset. Usage:
```python
import util
train_all            = util.load_train_data() # Load all train data
train_data, val_data = util.load_splitted_data() # Load and separate train data with validation data
test_data            = util.load_test_data() # Load test data
```
* &#x1F34E;&#x1F34E;&#x1F34E; To import util.py:
```python
import sys
sys.path.append(PATH_TO_REPO)
from src import util
```
* &#x1F34E;&#x1F34E;&#x1F34E; Danke schön
