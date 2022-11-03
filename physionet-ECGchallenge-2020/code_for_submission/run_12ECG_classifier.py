#!/usr/bin/env python

# command:  python evaluate_12ECG_score.py labels output scores.csv
# labels: D:\OneDrive\Desktop\CinC_project\data\test_balanced

import joblib
from keras.engine.saving import load_model
from preprocessor import preprocess_input
from main_model import MainModel
import tensorflow.keras as keras

def run_12ECG_classifier(data, header_data, classes, model):

    # pre-process input signals
    input_data = preprocess_input(data)

    # predict
    current_label, current_score = model.predict(input_data)

    return current_label[0], current_score[0]


def load_12ECG_model():
    # load the model from disk

    # Models stage 1
    stage_1_1 = load_model('models/stage_1_1.h5')  # CNN-LSTM
    stage_1_2 = keras.models.load_model('models/stage_1_2.h5')  # residual network

    # Model stage 2
    stage_2 = load_model('models/stage_2.h5')  # LSTM

    # Main model
    loaded_model = MainModel(stage_1_1, stage_1_2, stage_2)

    return loaded_model
