"""
Residual network model (stage_1_2)
"""
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold
import os
from utils import encode_labels, custom_multiclass_f1, multiclass_f1
import shutil
import ntpath
from collections import Counter
import numpy as np
from keras.initializers import glorot_uniform, he_normal
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays
from preprocessor import preprocess_input_data
from utils import encode_labels
import tensorflow.keras as keras
import pandas as pd
# import sys  # This gives an error on the CSC server
from model_architecture_V01 import Inc_ResNet_LSTM_v01


def cross_validation(arr_of_segments, arr_of_labels, arr_of_IDs, ids_labels):
    """ Subject cross-validation """

    global epochs, batch_size, n_folds

    # split the subjects on n folds keeping balance
    ids = ids_labels['subject']
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
    subj_folds = [(ids[test_index]) for train_index, test_index in skf.split(ids_labels['subject'],
                                                                             ids_labels['label']
                                                                             )]


    # true labels of each subject
    subject_labels = {ID: None for ID in list(ids)}
    for ID, label in zip(arr_of_IDs, arr_of_labels):
        subject_labels[ID[0]] = label

    # to save the predictions of each subject
    subject_predictions = {ID: [] for ID in list(ids)}

    # to save the f1-score of each fold
    scores = {}
    scores_custom = {}

    for i, validation_fold in enumerate(subj_folds):
        print(f"\n\nFold {i} ------------------------------------------------- \n")

        # selector
        selector = np.isin(arr_of_IDs.squeeze(), validation_fold)

        # validation
        arr_seg_validation = arr_of_segments[selector]
        arr_labels_validation = arr_of_labels[selector]
        arr_IDs_validation = arr_of_IDs[selector]

        # train
        arr_seg_train = arr_of_segments[np.invert(selector)]
        arr_labels_train = arr_of_labels[np.invert(selector)]
        arr_IDs_train = arr_of_IDs[np.invert(selector)]

        # TODO
        # Up-balance 'STE' (3x)
        add_to_input = []
        add_to_labels = []
        add_to_IDs = []
        for j in range(len(arr_labels_train)):
            if arr_labels_train[j][8] == 1:
                add_to_input.append(arr_seg_train[j])
                add_to_labels.append(arr_labels_train[j])
                add_to_IDs.append(arr_IDs_train[j])

        arr_seg_train_balanced = np.concatenate([add_to_input, arr_seg_train, add_to_input])
        arr_labels_train_balanced = np.concatenate([add_to_labels, arr_labels_train, add_to_labels])
        arr_IDs_train_balanced = np.concatenate([add_to_IDs, arr_IDs_train, add_to_IDs])

        # Build model
        model = Inc_ResNet_LSTM_v01(segment_size, 12, classes=9)

        # TODO
        # callbacks
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, verbose=0, mode='max')
        mcp_save = keras.callbacks.ModelCheckpoint(os.path.join(experiments_dir, experiment_name, f"model_fold_{i}.h5"),
                                                   save_best_only=True, monitor='val_categorical_accuracy', mode='max')
        reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=7, verbose=1,
                                                           epsilon=1e-4,
                                                           mode='max')

        # model.summary()

        model.fit(arr_seg_train_balanced, arr_labels_train_balanced, epochs=epochs, batch_size=batch_size,
                  verbose=1, validation_data=(arr_seg_validation, arr_labels_validation), shuffle=True,
                  callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

        # re-load best model
        del model
        model = keras.models.load_model(os.path.join(experiments_dir, experiment_name, f"model_fold_{i}.h5"))
        _, accuracy = model.evaluate(arr_seg_validation, arr_labels_validation, batch_size=batch_size, verbose=1)
        predictions = model.predict(arr_seg_validation, verbose=1)

        # print fold results
        print("Accuracy:", accuracy)

        f1_score, f1_score_list = multiclass_f1(arr_labels_validation, predictions, return_list=True)
        print("\nf1 score:", f1_score)
        print(f1_score_list)

        f1_score_custom, f1_score_custom_list = custom_multiclass_f1(arr_labels_validation, predictions,
                                                                     return_list=True)
        print("\nf1 score (custom):", f1_score_custom)
        print(f1_score_custom_list)

        # save predictions
        for ID, pred in zip(arr_IDs_validation, predictions):
            subject_predictions[ID[0]].append(pred)

        # save f1-score
        scores[f"fold_{i}"] = f1_score
        scores_custom[f"fold_{i}"] = f1_score_custom

        # save f1-score list (text file):
        with open(os.path.join(experiments_dir, experiment_name, "scores.txt"), 'a') as f:
            f.write(f"Fold {str(i)}:\n"
                    f"{str(f1_score_list)} (f1-score by class) \n"
                    f"{str(f1_score_custom_list)} (f1 score (custom) by class) \n")

    # Average f-1 score
    m, s = np.mean([v for v in scores.values()]), np.std([v for v in scores.values()])
    m_c, s_c = np.mean([v for v in scores_custom.values()]), np.std([v for v in scores_custom.values()])

    # save labels (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "subject_labels.npy"), subject_labels)

    # save predictions (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "subject_predictions.npy"), subject_predictions)

    # save f1-scores (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "scores.npy"), scores)
    np.save(os.path.join(experiments_dir, experiment_name, "scores_custom.npy"), scores_custom)

    print("\n==========================================================\n")
    print(f"CV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})")

    # save f1-scores (text file)
    with open(os.path.join(experiments_dir, experiment_name, "scores.txt"), 'a') as f:
        f.write("\n\n ==> Score by CV:")
        f.write(f"\n{str(scores)} (f1-score) \n{str(scores_custom)} (f1-score (custom))")
        f.write("\n\n ==> Average score CV:")
        f.write(f"\nCV f1-score: {str(m)} (+/- {str(s)}) \nCV f1-score (custom): {str(m_c)} (+/- {str(s_c)})\n\n")


if __name__ == '__main__':

    # Config
    experiment_name = "stage_1_3_002_Inc_ResNet_LSTM_v01"
    experiments_dir = "experiments_stage_1"
    data_dir = 'data/train_balanced'
    segment_size = 2000
    overlap = 0.5
    epochs = 50  # ???
    batch_size = 54  # ???
    n_folds = 8

    # create directory for the experiment
    if not os.path.exists(os.path.join(experiments_dir, experiment_name)):
        os.makedirs(os.path.join(experiments_dir, experiment_name))
    # else:
    #     raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
    #                     f" in the '{experiments_dir}' directory.")

    # # save a copy of the script
    # shutil.copy(__file__, os.path.join(experiments_dir, experiment_name, ntpath.basename(__file__)))

    # This gives an error on the CSC server when trying to import sys
    # # Log stdout
    # log_file = os.path.join(experiments_dir, experiment_name, 'logfile.log')
    # sys.stdout = Logger(log_file)

    # load data
    data = load_data(data_dir)

    # create array with the label of each subject ( it is used to keep the balance of the labels
    # in the folds of the cross-validation
    dic_labels = {}
    for k, v in data.items():
        dic_labels[k] = data[k]['info']['Dx']

    ids_labels = pd.Series(dic_labels).reset_index()
    ids_labels.columns = ['subject', 'label']

    # pre-process signals
    data = preprocess_input_data(data)

    # segment signal
    data = segment_all_dict_data(data, segment_size, overlap)

    arr_of_segments, arr_of_labels, arr_of_IDs = reshape_segmented_arrays(data,
                                                                          shuffle_IDs=True,
                                                                          # Do not shuffle the segments to keep the
                                                                          # order in time of the predictions
                                                                          shuffle_segments=False,
                                                                          segment_standardization_flag=True)

    # Encode labels
    arr_of_labels = np.array([i[0]['Dx'] for i in arr_of_labels])
    arr_of_labels = encode_labels(arr_of_labels)

    # Cross-validation
    cross_validation(arr_of_segments, arr_of_labels, arr_of_IDs, ids_labels)
