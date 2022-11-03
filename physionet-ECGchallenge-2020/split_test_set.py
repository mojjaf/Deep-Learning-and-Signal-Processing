"""
Script to split the data (Training_WFDB) in a train and test sets keeping the balance of the labels
"""
from collections import Counter
import numpy as np
from scipy.io import loadmat
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

fs = 500  # Hz
lead_labels = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_data(data_dir):
    """ Load the subject signals (".mat" files) and info (".hea" files)
        into a dictionary """

    # signals (".mat" files)
    data = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".mat"):
                print(file)

                # subject id
                subj_id = re.match(r"^(\w+).mat", file).group(1)
                # create dictionary for the subject
                data[subj_id] = {}

                # load signals from the mat file
                signals = loadmat(os.path.join(root, file))['val']

                # add the signals to the data dictionary
                for i in range(signals.shape[0]):
                    data[subj_id][lead_labels[i]] = signals[i, :]

    # labels and extra info (".hea" files)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hea"):
                print(file)

                # subject id
                subj_id = re.match(r"^(\w+).hea", file).group(1)

                with open(os.path.join(data_dir, file), 'r') as f:
                    text = f.read()

                # dictionary to save all the info about the subject
                info = {}

                # Age
                age = re.search(r"#Age: (.+?)\n", text).group(1)
                info['Age'] = age
                # Sex
                sex = re.search(r"#Sex: (.+?)\n", text).group(1)
                info['Sex'] = sex
                # Dx
                dx = re.search(r"#Dx: (.+?)\n", text).group(1)
                info['Dx'] = dx
                # Rx
                rx = re.search(r"#Rx: (.+?)\n", text).group(1)
                info['Rx'] = rx
                # Hx
                hx = re.search(r"#Hx: (.+?)\n", text).group(1)
                info['Hx'] = hx
                # Sx
                sx = re.search(r"#Sx: (.+?)\n", text).group(1)
                info['Sx'] = sx

                # add the info to the data dictionary
                data[subj_id]['info'] = info

    return data


data = load_data('data/alldata')
test_size = 0.2

dic = {}
for k, v in data.items():
    dic[k] = data[k]['info']['Dx']


arr = pd.Series(dic).reset_index()
arr.columns = ['subject', 'label']


X_train, X_test, y_train, y_test = train_test_split(arr['subject'], arr['label'], test_size=test_size, random_state=0, stratify=arr['label'])


label_count_train = Counter(y_train)
print(f"Balance test set: \n {label_count_train}")

label_count_test = Counter(y_test)
print(f"Balance test set: \n {label_count_test}")


for i in range(len(X_train)):
    name = X_train.iloc[i]

    mat_file = os.path.join('data/alldata', f"{name}.mat")
    header_file = os.path.join('data/alldata', f"{name}.hea")

    new_mat_file = os.path.join('data/train_balanced', f"{name}.mat")
    new_header_file = os.path.join('data/train_balanced', f"{name}.hea")

    shutil.move(mat_file, new_mat_file)
    shutil.move(header_file, new_header_file)

for i in range(len(X_test)):
    name = X_test.iloc[i]

    mat_file = os.path.join('data/alldata', f"{name}.mat")
    header_file = os.path.join('data/alldata', f"{name}.hea")

    new_mat_file = os.path.join('data/test_balanced', f"{name}.mat")
    new_header_file = os.path.join('data/test_balanced', f"{name}.hea")

    shutil.move(mat_file, new_mat_file)
    shutil.move(header_file, new_header_file)

print("DONE")
