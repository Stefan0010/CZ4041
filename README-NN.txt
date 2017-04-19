Dependencies:
- CUDA 8
- cuDNN v5.1
- Caffe framework with LSTM available
- Python 2.7
 +- NumPy
 +- pandas
 +- matplotlib
 +- SciPy

Files:
# NN for all store
- models/nn/pre.py: Preprocessing for NN
- models/nn/nn.py: Train and test code
- models/nn/nn.prototxt: NN model
- models/nn/nn_solv.prototxt: Caffe solver file

How to run NN for all store:
```
cd $PROJECT_DIR/models/nn
python pre.py && python nn.py
```
