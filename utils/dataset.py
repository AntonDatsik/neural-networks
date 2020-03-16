import os
import tarfile
import urllib.request

import matplotlib.image as mpimg
import numpy as np


def download_dataset(url):
    dataset_name, _ = os.path.splitext(os.path.basename(url))
    dir_path = os.path.join(os.getcwd(), 'data')
    os.makedirs(dir_path, exist_ok=True)

    full_path = os.path.join(dir_path, dataset_name)

    if not os.path.exists(full_path):
        urllib.request.urlretrieve(url, filename=full_path)

    return full_path


def extract_dataset(path):
    dir_name = 'dataset_{}'.format(os.path.basename(path).split(".")[0])
    dir_path = os.path.join(os.path.dirname(path), dir_name)

    if not os.path.exists(dir_path):
        with tarfile.open(path, 'r') as tar:
            tar.extractall(dir_path)

    return dir_path


def read_dataset(path, classes, img_width, img_height):
    n = img_height * img_width

    X = []
    Y = []

    for root, _, files in os.walk(path):
        for file in files:
            try:
                dir_name = os.path.basename(root)
                if dir_name in classes:
                    Y.append(classes.index(dir_name))

                    im = mpimg.imread(os.path.join(root, file))
                    X.append(im.reshape(1, n).T)
            except:
                pass

    m = len(X)
    X = np.array(X).T.reshape((n, m))
    Y = np.array(Y).T.reshape((1, m))

    return X, Y


def split_dataset(X, Y, train_size, valid_size, test_size):
    train_index = train_size
    valid_index = train_index + valid_size
    test_index = valid_index + test_size

    p = np.random.permutation(X.shape[1])

    X_split = np.hsplit(X[:, p], [train_index, valid_index, test_index])
    Y_split = np.hsplit(Y[:, p], [train_index, valid_index, test_index])
    return X_split[0], X_split[1], X_split[2], Y_split[0], Y_split[1], Y_split[2]
