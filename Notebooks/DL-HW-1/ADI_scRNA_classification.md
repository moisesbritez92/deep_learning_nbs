**Assignment**

1. Remember to add your name to the title of the notebook
2. The goal is to explore models that underfit and overfit, and to deal with overfitting by using the techniques seen in class.



```python
# Import needed libraries
import numpy as np
import sys, os, pdb
import pandas as pd
from matplotlib import pyplot as plt

```

Data:

Consists of the gene expression profile of several cells (coming from a patient). 

There is a train and a test datasets already provided to you.

They are organized as a matrix of cells x genes.

Given a cell, the goal is to predict the correct cell-type based on the genes' expressions for that sample.


```python
# Cambiar el directorio de trabajo a DL-HW-1 (se asume que `os` ya fue importado en otra celda)
target_dir = "DL-HW-1"

if os.path.isdir(target_dir):
    os.chdir(target_dir)
    print(f"Directorio cambiado a: {os.getcwd()}")
    print("Contenido del directorio:", os.listdir('.'))
else:
    raise FileNotFoundError(f"Directorio '{target_dir}' no existe. Ruta actual: {os.getcwd()}")
```


```python
# Load data

# Path to source batch
train_path = "train.pkl"
# Path to target batch
test_path = "test.pkl"
# Column containing cell-types
lname = "labels" 

train_batch = pd.read_pickle(train_path)
test_batch = pd.read_pickle(test_path)
```


```python
train_batch
```


```python
# Extract the common genes so that we can use the same network for both batches

common_genes = list(set(train_batch.columns).intersection(set(test_batch.columns)))
common_genes.sort()
train_batch = train_batch[list(common_genes)]
test_batch = test_batch[list(common_genes)]

train_mat = train_batch.drop(lname, axis=1)
train_labels = train_batch[lname]

test_mat = test_batch.drop(lname, axis=1)
test_labels = test_batch[lname]

# values are already normalized (ignore this)
mat = train_mat.values
mat_round = np.rint(mat)
error = np.mean(np.abs(mat - mat_round))

```


```python
train_labels.unique()
```

Q1: What type of problem are you dealing with?

Q2: What is the size of the input, i.e., number of features?

Q3: How many neurons should you have in the last layer?

Q4: What is the most appropiate activation function to use in the last layer? 


```python
 # Labels: process them to be in an adequate format

```


```python
Q5: How did you modify the labels?

Q6: Which loss function are you going to use to train the networks?
```


```python
# Minimum number of things you should try:

- 
```

Things I woul like to see (at least):

- Without using any kind of regularization, train 3 networks: one that underfits the data, one that does OK, and one that overfits very easily.
-- Comment on why you think each network behaves in a specific way: e.g., number of parameters, number of layers, size of each layer...
-- Comment on the results you observe

- Deal with overfitting by using:
-- L1, L2 or combination of both
-- Dropout
-- Comment on the results: what works better, how you chose the regularization parameter or dropout rate, etc.

- Show the results on the test set of your "best" model.

- Provide some conclusions of the analysis you performed.

