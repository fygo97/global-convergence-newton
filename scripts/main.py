import numpy as np
import requests
import urllib3
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import os
from logreg import CustomLogReg, MultivarLogReg
import matplotlib.pyplot as plt
from methods import LossFunction, Method, DataSet
import argparse

DATASET = DataSet.A9A

def make_plots(losses, accuracies, axs, row = 0): 
    axs[0].plot(losses)
    axs[0].set_xlabel("epochs")
    axs[0].set_ylabel("loss")
    axs[1].plot(accuracies)
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel("accuracy")


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
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    return X_train, y_train, X_test, y_test


def download_and_preprocess_ijcnn1():

    # Download data set
    if os.path.isfile("ijcnn1") == False:
        print("Couldn't find data sets... download ijcnn1 dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1")
        raise AssertionError("No data")

    # Load data
    X_train, y_train = load_svmlight_file("ijcnn1")
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)
    return X_train, y_train, X_test, y_test

def download_and_preprocess_mnist():

    # Download data set
    if os.path.isfile("ijcnn1") == False:
        print("Couldn't find data sets... download ijcnn1 dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#ijcnn1")
        raise AssertionError("No data")

    # Load data
    X_train, y_train = load_svmlight_file("mnist.scale")
    X_train = X_train.toarray()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':

    # Argparse 
    parser = argparse.ArgumentParser(description="Select dataset to use.")
    parser.add_argument("dataset", type=str, choices=["a9a", "covtype", "ijcnn1", "mnist"],
                        help="Dataset to use (a9a, covtype, ijcnn1, mnist)")
    parser.add_argument("loss", type=str, choices=["ce", "ncce"], 
                        help="Loss function to use (ce, ncce)")
    args = parser.parse_args()
    DATASET = args.dataset

    if DATASET == "a9a":
        X_train, y_train, X_test, y_test = download_and_preprocess_a9a()
    elif DATASET == "covtype":
        X_train, y_train, X_test, y_test = download_and_preprocess_covtype()
    elif DATASET == "ijcnn1":
        X_train, y_train, X_test, y_test = download_and_preprocess_ijcnn1()
    else:
        X_train, y_train, X_test, y_test = download_and_preprocess_mnist()
        print(y_test)

    if args.loss == "ce":
        loss_type = LossFunction.CE
    else:
        loss_type = LossFunction.NCCE

    print(f"number of samples = {len(y_train)}")
    print("x max/min:", np.max(X_train), np.min(X_train))
    print(type(X_test))
    print("Data sets have been loaded")

    # Training
    if DATASET != "mnist":
        y_train = np.clip(y_train, 0.0, 1.0)
        y_test = np.clip(y_test, 0.0, 1.0)
        lr = CustomLogReg(Method.NEWTON, loss_type=loss_type)
    else:
        lr = MultivarLogReg(Method.NEWTON, loss_type=loss_type)

    epochs = 6
    lr.fit(X_train, y_train, epochs=epochs, lr=0.1, batch_size=2048, lbd=0.0)
    print("Training complete")

    ones = np.ones(X_test.shape[0]).reshape((-1, 1))
    X_test = np.hstack([ones, X_test])
    print(X_train.shape)
    print(X_test.shape)
    pred = lr.predict(X_test)
    accuracy2 = accuracy_score(y_test, pred)

    # Plotting
    fig, axs = plt.subplots(1, 2)
    #make_plots(lr.losses, lr.train_accuracies, axs, row=0)
    make_plots(lr.losses, lr.train_accuracies, axs, row=0)
    #print(f"test accuracy GD: {accuracy}")
    print(f"test accuracy Newton: {accuracy2}")
    plt.show()
