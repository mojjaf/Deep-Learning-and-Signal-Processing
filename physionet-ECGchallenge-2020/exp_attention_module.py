from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays, split_train_test, \
    dict_to_array
from keras.layers import Dense, LSTM, Dropout, Convolution1D, LeakyReLU, BatchNormalization
from utils import encode_labels, custom_multiclass_f1, multiclass_f1
import shutil
import ntpath
from keras import Input, Model, initializers, regularizers, constraints
from keras.engine import Layer
from sklearn.model_selection import KFold
from preprocess_and_segmentation import load_data, segment_all_dict_data, reshape_segmented_arrays, split_train_test
from preprocessor import preprocess_input_data
import numpy as np
import os
from keras.layers import Dense, LSTM, Dropout, LeakyReLU, BatchNormalization, Bidirectional, ReLU, CuDNNGRU, GRU
from utils import encode_labels, custom_multiclass_f1, split_train_validation_part_2
from keras import backend as K
import tensorflow as tf


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,

                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name), shape=(input_shape[-1],),
                                     initializer='zero',

                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.u = self.add_weight(name='{}_u'.format(self.name), shape=(input_shape[-1],),
                                 initializer=self.init,

                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def build_model(n_timesteps, n_features, n_outputs):
    input = Input(shape=(n_timesteps, n_features), dtype='float32')

    x = Convolution1D(12, 3, padding='same')(input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)

    x = Bidirectional(LSTM(50, input_shape=(2225, 12), return_sequences=True, return_state=False))(cnnout)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Bidirectional(LSTM(50, input_shape=(n_timesteps, n_features), return_sequences=True, return_state=False))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    X = LeakyReLU(alpha=0.3)(x)
    output = Dense(n_outputs, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)

    # try:
    #     model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
    #     print("Training on 4 GPUs")
    # except:
    #     print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])

    return model


def cross_validation(arr_of_signals, arr_of_labels, arr_of_IDs):
    """ Subject cross-validation """

    global epochs, batch_size, n_folds

    # split the subjects on n folds
    ids_unique = np.unique(arr_of_IDs)
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=True)
    subj_folds = [(ids_unique[test_index]) for train_index, test_index in kf.split(ids_unique)]

    # true labels of each subject
    subject_labels = {ID: [] for ID in list(ids_unique)}
    for ID, label in zip(arr_of_IDs, arr_of_labels):
        subject_labels[ID[0]] = label

    # to save the predictions of each subject
    subject_predictions = {ID: [] for ID in list(ids_unique)}

    # to save the f1-score of each fold
    scores = {}
    scores_custom = {}

    for i, validation_fold in enumerate(subj_folds):
        print(f"\n\nFold {i} ------------------------------------------------- \n")

        # TODO ------
        if i >= 1:
            break
        # TODO ------

        # selector
        selector = np.isin(arr_of_IDs.squeeze(), validation_fold)

        # validation
        arr_seg_validation = arr_of_signals[selector]
        arr_labels_validation = arr_of_labels[selector]
        arr_IDs_validation = arr_of_IDs[selector]

        # train
        arr_seg_train = arr_of_signals[np.invert(selector)]
        arr_labels_train = arr_of_labels[np.invert(selector)]
        arr_IDs_train = arr_of_IDs[np.invert(selector)]

        #  Model
        n_timesteps, n_features, n_outputs = arr_of_signals.shape[1], arr_of_signals.shape[2], arr_of_labels.shape[1]

        model = build_model(n_timesteps, n_features, n_outputs)
        model.summary()

        model.fit(arr_seg_train, arr_labels_train, epochs=epochs, batch_size=batch_size,
                  verbose=1, validation_data=(arr_seg_validation, arr_labels_validation), shuffle=True)

        _, accuracy = model.evaluate(arr_seg_validation, arr_labels_validation, batch_size=batch_size, verbose=1)
        predictions = model.predict(arr_seg_validation, verbose=1)

        # print fold results
        print("Accuracy:", accuracy)
        f1_score = multiclass_f1(arr_labels_validation, predictions)
        print("fi score (custom):", f1_score, "\n\n")
        f1_score_custom = custom_multiclass_f1(arr_labels_validation, predictions)
        print("fi score (custom):", f1_score_custom, "\n\n")

        # save predictions
        for ID, pred in zip(arr_IDs_validation, predictions):
            subject_predictions[ID[0]].append(pred)

        # save f1-score
        scores[f"fold_{i}"] = f1_score
        scores_custom[f"fold_{i}"] = f1_score_custom

        # save model (to disk)
        model.save(os.path.join(experiments_dir, experiment_name, f"model_fold_{i}.h5"))

    # Average f-1 score
    m, s = np.mean([v for v in scores.values()]), np.std([v for v in scores.values()])
    m_c, s_c = np.mean([v for v in scores_custom.values()]), np.std([v for v in scores_custom.values()])

    print("\n==========================================================\n")
    print(f"CV f1-score: {str(m)} (+/- {str(s)}) \n CV f1-score (custom): {str(m_c)} (+/- {str(s_c)})")

    # save f1-score
    with open(os.path.join(experiments_dir, experiment_name, "scores.txt"), 'w') as f:
        f.write(f"CV f1-score: {str(m)} (+/- {str(s)}) \n CV f1-score (custom): {str(m_c)} (+/- {str(s_c)})")

    # save labels (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "subject_labels.npy"), subject_labels)

    # save predictions (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "subject_predictions.npy"), subject_predictions)

    # save f1-scores (to disk)
    np.save(os.path.join(experiments_dir, experiment_name, "scores.npy"), scores)


if __name__ == '__main__':

    # Config
    experiment_name = "exp_winner_model_attention"
    experiments_dir = "experiments_part_1"
    data_dir = 'sample_of_data/Training_WFDB'  # TODO
    fs = 500  # Hz
    # segment_size = 2999
    # overlap = 0.5
    epochs = 15
    batch_size = 34
    n_folds = 8  # n of folds in the cross_validation

    # create directory for the experiment
    if not os.path.exists(os.path.join(experiments_dir, experiment_name)):
        os.makedirs(os.path.join(experiments_dir, experiment_name))
    else:
        raise NameError(f"Already exist an experiment with the name '{experiment_name}'"
                        f" in the '{experiments_dir}' directory.")

    # save a copy of the script
    shutil.copy(__file__, os.path.join(experiments_dir, experiment_name, ntpath.basename(__file__)))

    # load data
    data = load_data(data_dir)

    # pre-process signals
    data = preprocess_input_data(data, standardization_flag=True, padding_flag=True)

    arr_of_signals, arr_of_labels, arr_of_IDs = dict_to_array(data, shuffle_IDs=True)

    # Encode labels
    arr_of_labels = np.array([i[0]['Dx'] for i in arr_of_labels])
    arr_of_labels = encode_labels(arr_of_labels)

    # Cross-validation
    cross_validation(arr_of_signals, arr_of_labels, arr_of_IDs)