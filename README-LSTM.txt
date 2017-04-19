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
# LSTM per store model #1
- models/lstm_per_store/prev2.py: preprocessing
- models/lstm_per_store/lstmv2.py: Train and test code
- models/lstm_per_store/lstmv2.prototxt: Caffe network model file
- models/lstm_per_store/solverv2.prototxt: Caffe solver file

# LSTM per store model #2 (Better result)
- models/lstm_per_store/prev2_1.py: preprocessing
- models/lstm_per_store/lstmv2_1.py: Train and test code
- models/lstm_per_store/lstmv2_1.prototxt: Caffe network model file
- models/lstm_per_store/solverv2_1.prototxt: Caffe solver file

How to run LSTM per store #1:
```
cd $PROJECT_DIR/models/lstm_per_store
rm -rf temp
mkdir temp
python prev2.py && python lstmv2.py
```

How to run LSTM per store #2:
```
cd $PROJECT_DIR/models/lstm_per_store
rm -rf temp
mkdir temp
python prev2_1.py && python lstmv2_1.py
```
