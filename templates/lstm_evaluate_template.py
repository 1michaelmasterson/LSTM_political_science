########################################
# Library packages
########################################
import pandas as pd
import numpy as np
import keras
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

########################################
# fix random seed for reproducibility
########################################
seed = 34324  # your seed here
np.random.seed(seed)

########################################
# set path(s) to load data and save results
########################################

train_file = "../data/dataset.csv"  # path and name of your dataset goes here

#  results go here
results_path = "../results/scores/"  # your filepath here
results_file_name = "results.csv"  # yourfilename here

########################################
# Set hyper-parameters
########################################
# you should supply your own hyper-parameters based on your model
# tuning results to replace the example hyper-parameters below

# set the size of the embedding vector
embedding_vecor_length = 100

# set number of hidden units in lstm
hn = 20

# set the max number of epochs to train
nb_epoch = 100

# set batch size
batch_size = 200

# set your learning rate
lr = 0.001

# set the proportion of your data to use as the test set
test_size = 0.1

# set the proportion of your remaining training data to use as a validation set
val_size = 0.1

# set callbacks and early stopping to stop after 2 epochs
# of no improvement in validation accuracy
my_callbacks = [
    EarlyStopping(monitor='val_acc', patience=2, verbose=1, mode='max')
]
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


########################################
# Make function that defines the model
########################################


def create_model(X, y, X_test, y_test):
    """This function defines an LSTM Model. If you wish to change the architecture
    of the model, then you can edit this function. Otherwise,
     we recommend you just adjust the hyper-parameters"""
    # clears memory before model trains each time
    K.clear_session()

    # makes validation set you can adjust its size as needed
    X, X_val, y, y_val = train_test_split(X,
                                          y,
                                          test_size=val_size,
                                          random_state=seed,
                                          stratify=y)

    # This tokenizer assumes your text has already been preprocessed
    TK = Tokenizer(filters='', lower=True, split=" ")

    # Convert all documents into sequences
    TK.fit_on_texts(list(X))
    X = TK.texts_to_sequences(X)
    X_test = TK.texts_to_sequences(X_test)
    X_val = TK.texts_to_sequences(X_val)

    # This tells you how many words are in your training set
    top_words = len(TK.word_index) + 1
    print('Total words: %d' % top_words)

    # Pad all sequences to the same length
    X = sequence.pad_sequences(X,
                               maxlen=max_doc_length + 1,
                               padding='post',
                               truncating='post')
    X_test = sequence.pad_sequences(X_test,
                                    maxlen=max_doc_length + 1,
                                    padding='post',
                                    truncating='post')
    X_val = sequence.pad_sequences(X_val,
                                   maxlen=max_doc_length + 1,
                                   padding='post',
                                   truncating='post')

    ########################################
    # #Now define the model
    ########################################
    model = Sequential()

    # add embedding layer to model
    model.add(
        Embedding(top_words,
                  embedding_vecor_length,
                  mask_zero=True,
                  input_length=max_doc_length + 1))

    # Dropout the embedding layer to prevent overfitting
    model.add(Dropout(0.5))

    # add bidirectional LSTM layer to model
    # with dropout and recurrent dropout to prevent overfitting
    model.add(
        Bidirectional(LSTM(hn, dropout=0.5, recurrent_dropout=0.5,
                           unroll=True)))

    # Add final layer that will issue a binary classification
    model.add(Dense(1, activation='sigmoid'))

    # specify the optimizer
    optimizer = keras.optimizers.RMSprop(lr=lr)

    # Compile the model and print its summary
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', recall, precision])
    print(model.summary())

    # fit the model and keep track of score history
    model.fit(X,
              y,
              callbacks=my_callbacks,
              epochs=100,
              batch_size=200,
              validation_data=(X_val, y_val))

    # Evaluate model performance
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    cvscores.append(scores[1])
    cvrecall.append(scores[2])
    cvpre.append(scores[3])

    return scores


########################################
# load the dataset
########################################
# Here we assume your data has already been preprocessed (or not)
# according to your preference
# this script expects a csv file with a category column that indicates
# the 2 catories you data takes
# with numeric lables 1 or 0
# and a text column that contains the text
# it is okay if your dataset contains other columns
# they will be dropped before analysis continues

# load the data as a pandas dataframe called train
dtypes = {'category': 'int', 'text': 'str'}
df = pd.read_csv(train_file, dtype=dtypes)

# drop excess columns if any
selected = ['category', 'text']
non_selected = list(set(df.columns) - set(selected))
df = df.drop(non_selected, axis=1)
df = df.dropna(axis=0, how='any', subset=selected)
df = df.reindex(np.random.permutation(df.index))

########################################
# Split data into a features and labels vectors
########################################
# grabs the text
X_train = df[selected[1]]

# grabs the labels
y_train = pd.Series(df[selected[0]])

########################################
# Find length of longest document and set LSTM to use it
########################################
# by default this will use the full length of all of your documents
# if your documents are too long, uncomment the line below to
# manually set a length to shorten documents to
max_doc_length = max([len(x.split(" ")) for x in X_train])

# max_doc_length = insert_shorter_length_here

########################################
# Cross validate
########################################
# create an object that will split your data for cross validiation
# you can adjust the number of splits and size of the test dataset as needed
# StratifiedShufflesplit will try to split your data into
# as balanced subsamples as possible
kfold = StratifiedShuffleSplit(n_splits=5,
                               test_size=test_size,
                               random_state=seed)

# create lists to append results to
cvscores = []
cvrecall = []
cvpre = []

# Crossvalidate the model
for train, test, in kfold.split(X_train, y_train):
    # print(y_train[train])
    create_model(X=X_train[train],
                 y=y_train[train],
                 X_test=X_train[test],
                 y_test=y_train[test])

########################################
# Save results
########################################
data = list(zip(cvscores, cvrecall, cvpre))
results_df = pd.DataFrame(data=data, columns=['acc', 'recall', 'prec'])

results_df.to_csv((results_path) + str(results_file_name), index=False)
