import numpy as np
import requests
import urllib3
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import os
from Logreg import CustomLogReg
import matplotlib.pyplot as plt
from methods import Method

DATASET = "ijcnn1"

def make_plots(losses, accuracies, axs, row = 0):
    xs = range(len(losses))
    axs[row, 0].plot(losses)
    axs[row, 0].set_xlabel("epochs")
    axs[row, 0].set_ylabel("loss")
    axs[row, 1].plot(accuracies)
    axs[row, 1].set_xlabel("epochs")
    axs[row, 1].set_ylabel("accuracy")


def download_and_preprocess_a9a():

    # Download data set
    if os.path.isfile("a9a.t") == False or os.path.isfile("a9a") == False:
        print("Couldn't find data sets... attempting to download")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
        response = requests.get(url, verify=False) # Verify set to false (not very safe but works)
        with open("a9a", "wb") as f:
            f.write(response.content)
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
        response = requests.get(url, verify=False) # Verify set to false (not very safe but works)
        with open("a9a.t", "wb") as f:
            f.write(response.content)

    # Load data
    X_train, y_train = load_svmlight_file("a9a")
    X_test, y_test = load_svmlight_file("a9a.t", n_features=X_train.shape[1])

    # Convert from sparse to dense, then cast to float32
    X_train = X_train.toarray().astype(np.float32)
    X_test = X_test.toarray().astype(np.float32)

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return X_train, y_train, X_test, y_test

def download_and_preprocess_covtype():

    # Download data set
    if os.path.isfile("covtype") == False:
        print("Couldn't find data sets... attempting to download")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2"
        response = requests.get(url, verify=False) # Verify set to false (not very safe but works)
        with open("covtype", "wb") as f:
            f.write(response.content)

    # Load data
    X_train, y_train = load_svmlight_file("covtype")

    X_train = X_train.toarray().astype(np.float32)
    y_train = y_train.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)
    X_test = np.array(X_test, dtype=np.float32)

    return X_train, y_train, X_test, y_test


def download_and_preprocess_ijcnn1():

    # Download data set
    if os.path.isfile("ijcnn1") == False:
        print("Couldn't find data sets... download ijcnn1 dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1")
        raise AssertionError("No data")

    # Load data
    X_train, y_train = load_svmlight_file("ijcnn1")
    X_train = X_train.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)
    X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    if DATASET == "a9a":
        X_train, y_train, X_test, y_test = download_and_preprocess_a9a()
    elif DATASET == "covtype":
        X_train, y_train, X_test, y_test = download_and_preprocess_covtype()
    else:
        X_train, y_train, X_test, y_test = download_and_preprocess_ijcnn1()


    print(f"number of samples = {len(y_train)}")
    print("x max/min:", np.max(X_train), np.min(X_train))
    print(type(X_test))
    print("Data sets have been loaded")

    # Prepocessing
    y_train = np.clip(y_train, 0.0, 1.0)
    y_test = np.clip(y_test, 0.0, 1.0)

    # Gradient Descent
    #epochs = 150
    #lr = CustomLogReg(Method.GD)
    # lr.fit(X_train, y_train, epochs=epochs, lr=1, batch_size=y_train.shape[0])
    #print("Training complete")
    #pred = lr.predict(X_test)
    #accuracy = accuracy_score(y_test, pred)

    # Newton's method
    epochs = 50
    lr2 = CustomLogReg(Method.NEWTON)
    lr2.fit(X_train, y_train, epochs=epochs, lr=0.1, batch_size=2048, lbd=1e-3)
    print("Training complete")
    pred = lr2.predict(X_test)
    accuracy2 = accuracy_score(y_test, pred)

    #plotting
    fig, axs = plt.subplots(2, 2)
    #make_plots(lr.losses, lr.train_accuracies, axs, row=0)
    make_plots(lr2.losses, lr2.train_accuracies, axs, row=1)
    #print(f"test accuracy GD: {accuracy}")
    print(f"test accuracy Newton: {accuracy2}")
    plt.show()
