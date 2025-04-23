# SQS: Efficient Bayesian DNN Compression through Sparse Quantized Sub-distributions



## 1. Prerequisite of using these methods

- install the dependency package

```bash
pip install -r requirements.txt
```

- the base code library of model compression and quatization:
```bash
cd src/SQS
pip install -e .
```

## 2. Directory

### Base model

- `src/BERT`: use the BERT neural network as the base model.
- `src/GLUE`: the data oracle API to draw data. 
- `src/GPT2`:  use the GPT2 neural network as the base model.
- `src/resnet`: use the residual network as the base model.

### Methods

- `src/SQS/SQS/modelling/DGMS`: the DGMS baseline method.
- `src/SQS/SQS/modelling/DGMS/GMM.py`: the proposed method in this work.



### 3. Look at the summarized result
The experimental results are summarized in the `out` folders.
