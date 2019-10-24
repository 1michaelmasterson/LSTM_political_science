# The results and parameters of the
# best performing model will print to the terminal when finished
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
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from scipy.stats import uniform
from keras.wrappers.scikit_learn import KerasClassifier

########################################
# fix random seed for reproducibility
########################################
seed = 34324  # your seed here
np.random.seed(seed)

########################################
# set path(s) to load data and save results
########################################

train_file = "../data/dataset.csv"  # path and name of your dataset goes here

########################################
# Define hyper-parameters to search over
########################################
# How many random search should be run?
n_iter = 2  # keep in mind each iteration is 5 cross-validations

# Set uniform distribution of learning rates to search over
# loc is the minimum of the distribution and loc + scale is the maximum
# size is the number of draws from the distribution
lr = list(uniform.rvs(loc=0.00055, scale=0.0075, size=n_iter))

# This will return random integer between low and high
# to try as values for the number of hidden units
hn = list(np.random.randint(low=5, high=140, size=n_iter))

########################################
# Set hyper-parameters you will not search over
########################################
# These hyper-parameters are fixed and will not be tuned

embedding_vecor_length = 100

# set the max number of epochs to train
nb_epoch = [100]

# set batch size
batch_size = [200]

# set the proportion of your data to use as the test set
test_size = 0.1

# set the proportion of your remaining training data to use as a validation set
val_size = 0.1

# set callbacks and early stopping to stop after 2 epochs of
# no improvement in validation accuracy
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


def docs_to_sequence(docs):
    # This tokenizer assumes your text has already been preprocessed
    TK = Tokenizer(filters='', lower=True, split=" ")

    # Convert all documents into sequences
    TK.fit_on_texts(list(docs))
    docs = TK.texts_to_sequences(docs)

    # This tells you how many words are in your training set
    top_words = len(TK.word_index) + 1
    print('Total words: %d' % top_words)

    # Pad all sequences to the same length
    docs = sequence.pad_sequences(docs,
                                  maxlen=max_doc_length + 1,
                                  padding='post',
                                  truncating='post')

    return docs, top_words


# keep track of progress
count = 0


def create_model(hn, lr, epochs, batch_size):
    """This function defines an LSTM Model. If you wish to change the architecture
    of the model, then you can edit this function. Otherwise,
     we recommend you just adjust the hyper-parameters"""
    # clears memory before model trains each time
    K.clear_session()

    ########################################
    # Now define the model
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
    # show progress
    global count
    if count == 1:
        print(model.summary())
    print(str(count) + ' out of ' + str(n_iter * 5))
    count = count + 1

    return model


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
# turn docs to sequences
########################################
X_train, top_words = docs_to_sequence(X_train)

########################################
# TUNE
########################################

model = KerasClassifier(build_fn=create_model, verbose=1)

param_grid = dict(lr=lr, epochs=nb_epoch, batch_size=batch_size, hn=hn)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=7)
grid = RandomizedSearchCV(estimator=model,
                          param_distributions=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          n_iter=n_iter,
                          return_train_score=False,
                          fit_params={
                              'callbacks': my_callbacks,
                              'validation_split': 0.1
                          })

grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" %
      (grid_result.best_score_, grid_result.best_params_))
