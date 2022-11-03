"""
Residual network model (stage_1_2)
"""

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


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=1, strides=1, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c')(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    ### END CODE HERE ###

    return X


def maxpool_block_1(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=f, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Second component of main path ()
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '2b',
                            kernel_initializer=he_normal(seed=0))(
        X)

    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same', name=mp_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    ### END CODE HERE ###

    return X


def maxpool_block_2(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F1, kernel_size=f, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2b', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)
    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c',
                            kernel_initializer=he_normal(seed=0))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F2, kernel_size=1, strides=1, name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)

    X_shortcut = keras.layers.MaxPooling1D(pool_size=s, padding='same', name=mp_name_base + '1')(X_shortcut)
    # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    ##### MAIN PATH #####
    # First component of main path
    X = keras.layers.Conv1D(filters=F1, kernel_size=3, strides=s, padding='same',
                            input_shape=(None, n_timesteps, n_features),
                            name=conv_name_base + '2a', kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Second component of main path (≈3 lines)
    X = keras.layers.Conv1D(filters=F2, kernel_size=24, strides=1, padding='same', name=conv_name_base + '2b',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    # Third component of main path (≈2 lines)
    X = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=1, padding='same', name=conv_name_base + '2c',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.2)(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = keras.layers.Conv1D(filters=F3, kernel_size=1, strides=s, padding='same', name=conv_name_base + '1',
                                     kernel_initializer=he_normal(seed=0))(X_shortcut)
    # print(X_shortcut.shape)
    X_shortcut = keras.layers.BatchNormalization(name=bn_name_base + '1')(X_shortcut)
    # X_shortcut = Dropout(0.25)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = keras.layers.add([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


def ResNet1D(input_shape=(2000, 12), classes=9):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = keras.layers.Input(input_shape)
    # Zero-Padding
    X = keras.layers.ZeroPadding1D(0)(X_input)

    # Stage 1
    X = keras.layers.Conv1D(filters=16, kernel_size=7, padding='same', name='conv1',
                            kernel_initializer=he_normal(seed=0))(X)
    X = keras.layers.BatchNormalization(name='bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)

    # Stage 2

    X = maxpool_block_1(X, f=5, filters=[16, 32], s=1, stage=2, block='a')
    X = identity_block(X, 7, [16, 16, 32], stage=2, block='b')
    X = identity_block(X, 7, [16, 16, 32], stage=2, block='c')
    X = convolutional_block(X, f=2, filters=[16, 16, 32], s=1, stage=2, block='d')

    # Stage 3

    X = maxpool_block_2(X, 5, filters=[16, 32], s=2, stage=3, block='a')
    X = maxpool_block_2(X, 13, filters=[16, 32], s=1, stage=3, block='b')
    X = maxpool_block_2(X, 13, filters=[16, 32], s=2, stage=3, block='d')
    X = identity_block(X, 7, [16, 16, 32], stage=3, block='e')
    X = identity_block(X, 7, [16, 16, 32], stage=3, block='f')
    X = convolutional_block(X, f=2, filters=[16, 16, 32], s=1, stage=3, block='g')

    # Stage 4
    X = maxpool_block_2(X, 5, filters=[32, 64], s=1, stage=4, block='a')
    X = maxpool_block_2(X, 13, filters=[32, 64], s=2, stage=4, block='b')
    X = maxpool_block_2(X, 13, filters=[32, 64], s=2, stage=4, block='d')
    X = identity_block(X, 7, [32, 32, 64], stage=4, block='e')
    X = identity_block(X, 7, [32, 32, 64], stage=4, block='f')
    X = convolutional_block(X, f=2, filters=[32, 32, 64], s=1, stage=4, block='g')

    # Stage 5
    X = maxpool_block_2(X, 5, filters=[64, 128], s=1, stage=5, block='a')
    X = maxpool_block_2(X, 13, filters=[64, 128], s=2, stage=5, block='b')
    X = maxpool_block_2(X, 13, filters=[64, 128], s=2, stage=5, block='d')
    X = identity_block(X, 7, [64, 64, 128], stage=5, block='e')
    X = identity_block(X, 7, [64, 64, 128], stage=5, block='f')
    X = convolutional_block(X, f=2, filters=[64, 64, 128], s=1, stage=5, block='g')

    # Stage 6
    X = maxpool_block_2(X, 5, filters=[128, 256], s=1, stage=6, block='a')
    X = maxpool_block_2(X, 13, filters=[128, 256], s=2, stage=6, block='b')
    X = maxpool_block_2(X, 13, filters=[128, 256], s=2, stage=6, block='d')
    X = identity_block(X, 7, [128, 128, 256], stage=6, block='e')
    X = identity_block(X, 7, [128, 128, 256], stage=6, block='f')
    X = convolutional_block(X, f=2, filters=[128, 128, 256], s=1, stage=6, block='g')

    X = keras.layers.BatchNormalization(name='bn_final')(X)
    X = keras.layers.Activation('relu')(X)

    # X=LSTM(50, return_sequences=True,input_shape=(X.shape[1],1))(X)

    # X=LSTM(20)(X)
    # X = MaxPooling1D(pool_size=2, name='max_pool')(X)
    # X = TimeDistributed(Flatten())(X)
    X = keras.layers.Flatten()(X)
    # X = LSTM(100)(X)
    # X = CuDNNLSTM(1000)(X)

    #
    # X=Dense(10, activation='relu',activity_regularizer=l1(0.0001), kernel_regularizer=regularizers.l2(0.001))(X)
    X = keras.layers.Dense(150, activation='relu')(X)
    X = keras.layers.Dropout(0.3)(X)

    X = keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X, name='ResNet1D')

    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model


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
        model = ResNet1D(input_shape=(segment_size, 12), classes=9)

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
    experiment_name = "stage_1_2_001_baseline"
    experiments_dir = "experiments_stage_1"
    data_dir = 'data/train_balanced'
    segment_size = 2000
    overlap = 0.5
    epochs = 50
    batch_size = 54
    n_folds = 8

    # create directory for the experiment
    if not os.path.exists(os.path.join(experiments_dir, experiment_name)):
        os.makedirs(os.path.join(experiments_dir, experiment_name))
    else:
        raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
                        f" in the '{experiments_dir}' directory.")

    # save a copy of the script
    shutil.copy(__file__, os.path.join(experiments_dir, experiment_name, ntpath.basename(__file__)))

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
