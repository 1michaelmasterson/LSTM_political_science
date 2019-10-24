# LSTM_political_science


This repository contains code the templates to apply the LSTM models from "Using Word Order in Political Text Classification with Long Short-term Memory Models," which is forthcoming in *Political Analaysis*, to your own work. Please cite the paper if you make use of anything in this repository. If you are looking instead for the replication materials, they are available here https://doi.org/10.7910/DVN/MRVKIR.

# Table of contents
1. Prerequisites
2. Directory Overview
3. Code Templates

# 1 Prerequisites
The Python scripts should be able to run on Windows, Mac, or Linux provided that you install tensorflow 1.3.0 from the wheel that is appropriate to your operating system (see wheel download instructions below). It was originally run on a computer using the Ubuntu 16.04 LTS operating system. To run the code in this repository, you need both Python3 and R installed. The code assumes that you have kept the structure of the repository intact. Otherwise, you may need to change file paths. This code also requires certain libraries installed for both R and Python.

The following R libraries are required: here, ggplot2, and dplyr.

The following Python3 libraries are required (other versions of the libraries referenced here may work but these are the versions it has been tested with):

gensim==2.3.0

h5py==2.8.0

jieba==0.39

Keras==2.2.2

Keras-Applications==1.0.4

Keras-Preprocessing==1.0.2

numpy==1.13.3

pandas==0.24.1

scikit-learn==0.20.2

scipy==0.19.1

sklearn==0.20.2

tensorflow-gpu==1.3.0 (or equivalent cpu version) This is an older version of tensorflow, so you will need to download the                         approriate 1.3.0 wheel for your system	from https://github.com/mind/wheels and install tensorflow from                       that wheel. If you use the GPU version, ensure you have
                      installed the dependencies including Cuda compilation tools, release 8.0, V8.0.61. 

tensorflow-tensorboard==0.1.8

# 2 Directory Overview
The data folder is where you should add your own data.

The templates folder contains the code templates to facilitate the application of LSTM in Political Science. First choose the template that is appropriate to your task. 

The results folder is empty until the python scripts are run. Each script will save its results to the scores sub-folder within the results folder. The predictions and models folder both start empty and are used to save model and prediction output from the code templates.

# 3 Code Templates
This repository contains code templates that you can use to apply LSTM models to your own data. There are 4 templates in the template folder. They are heavily commented to explain the portions that you need to adjust vs. what you can leave alone. Typically, you will need to provide the path to your data as well as the hyper-parameters or hyper-parameter ranges. The portions of the script that you will need to adjust are towards the top of the script. The only exception to this is if you want to truncate your documents to below their maximum length. In this case, scroll down to max_doc_length and set it equal to the value you desire. All of the scripts assume that your texts have already been pre-processed. Your data should be in csv format with the texts of the documents in a column called text and the categories of documents (labeled either 1 or 0) in a column called category. These scripts are designed for binary classification.

lstm_tune_hyper-params.py will tune hyper-parameters within ranges that you supply on data that you supply.

lstm_evaluate_template.py will do a 5 draw cross-validation of an lstm model with the hyper-parameters and data you supply.

lstm_fully_train_and_save_model.py will fully train and save an lstm model with the hyper-parameters and data you supply.

lstm_load_and_predict.py will load a model saved using the previous script to predict on a new set of data you supply.



**If you wish to use more than 2 categories**

Change the number of hidden units in the final dense layer to the number of categories you have and change the loss function to categorical_crossentropy or sparse_categorical_crossentropy depending on whether your labels are categorical targets or integer targets respectively. See https://keras.io/losses/ for more information.




