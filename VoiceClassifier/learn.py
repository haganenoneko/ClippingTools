from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor 

from keras.callbacks import EarlyStopping 

from datetime import datetime
import numpy as np 

# ---------------------------------------------------------------------------- #
#            Split data into 70% train, 20% validation, and 10% test           #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                      Scale features using StandardScaler                     #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#         Build categorical classifier with early stopping and softmax         #
# ---------------------------------------------------------------------------- #

"""
Dense implements the operation: 
    output = activation(dot(input, kernel) + bias) 

where 
- activation is the element-wise activation function passed as the activation argument, 
- kernel is a weights matrix created by the layer, and 
- bias is a bias vector created by the layer (only applicable if use_bias is True). 
https://keras.io/api/layers/core_layers/dense/

The Dropout layer randomly sets input units to 0 with a frequency of `rate` at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
https://keras.io/api/layers/regularization_layers/dropout/
"""

# ---------------------------------------------------------------------------- #
#                                 Fit the model                                #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                            Show model diagnostics                            #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Predict                                   #
# ---------------------------------------------------------------------------- #

