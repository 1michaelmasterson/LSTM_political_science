########################################
# Library packages
########################################
import pandas as pd
import numpy as np
import keras
# from keras.layers import LSTM
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout
# from keras.layers.wrappers import Bidirectional
# from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
from keras import backend as K
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedShuffleSplit
# Need pickle to save tokenizer
import pickle

########################################
# fix random seed for reproducibility
########################################
seed = 34324  # your seed here
np.random.seed(seed)

########################################
# set path(s) to load data and save results
########################################

predict_file = "../data/dataset.csv"  # path and name of your dataset goes here

# The model goes here and its tokenizer goes in a subfolder called tokenizers
# Note that to save a model you need libhdf5-dev and h5py installed
# If you wish to save a trained model uncomment the following line
# and supply a path
model_name = "model"  # your model name here
model_path = "../results/models/"  # your filepath here

# results go here
results_path = "../results/predictions/"  # your filepath here
results_file_name = "predictions.csv"  # yourfilename here

########################################
# Manually set max document length
########################################
# you must set the maximium document length to be the same as the max length
# the model you are loading trained on
# If you used the templates to train your model this will be the maximum
# document length +1 because an extra padded 0 is added
max_doc_length = 59  # your maximum document length here

########################################
# Manually set batch size
########################################
# The batch_size for the predictions
# must match the batch size you trained with

batch_size = 200  # your batch size here

########################################
# Make class to get loss history
########################################
# You can probably leave this section be unless there are more metrics
# you wish to add


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.recall = []
        self.val_recall = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.recall.append(logs.get('recall'))
        self.val_recall.append(logs.get('val_recall'))


# Define Evaluation Metrics
def recall(y_true, y_pred):
    """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# add metrics where Keras expects them
keras.metrics.recall = recall
keras.metrics.precision = precision

########################################
# load the dataset
########################################
# Here we assume your data has already been preprocessed (or not)
# according to your preference
# this script expects a text column that contains the text
# it is okay if your dataset contains other columns
# they will be dropped before analysis continues

# load the data as a pandas dataframe called train
dtypes = {'text': 'str'}
df = pd.read_csv(predict_file, dtype=dtypes)

# drop excess columns if any
selected = ['text']
non_selected = list(set(df.columns) - set(selected))
df = df.drop(non_selected, axis=1)
df = df.dropna(axis=0, how='any', subset=selected)
df = df.reindex(np.random.permutation(df.index))

########################################
# Load Model
########################################
print('loading model')
model = keras.models.load_model(model_path + model_name + ".hd5")

with open(model_path + "tokenizers/" + model_name + ".pickle", 'rb') as handle:
    TK = pickle.load(handle)

########################################
# Get features
########################################
# grabs the text
X_predict = df[selected[0]]

########################################
# Process features
########################################

X_predict = TK.texts_to_sequences(X_predict)

X_predict = sequence.pad_sequences(X_predict,
                                   maxlen=max_doc_length,
                                   padding='post',
                                   truncating='post')

########################################
# Make predictions
########################################
print('starting predictions')

# use to predict classes
predictions = model.predict_classes(X_predict,
                                    batch_size=batch_size,
                                    verbose=1)

# use to predict probabilities
# predictions = model.predict_proba(x, batch_size=batch_size, verbose=1)

########################################
# Save results
########################################
df['predictions'] = predictions

df.to_csv(results_path + str(results_file_name), index=False)
