import logging
import numpy as np
import requests
import urllib3
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import os
from logreg import MultivarLogReg
import matplotlib.pyplot as plt
from methods import LossFunction, Method
import argparse
logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='logreg_training.log',  # Log file name
    level=logging.INFO,             # Minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'                     # Overwrite log file each run; use 'a' to append
)


def make_plots(losses, accuracies, grad_norm, axs, row=0): 
    axs[0].plot(losses)
    axs[0].set_ylabel("loss")
    axs[0].set_xlabel("epochs")

    axs[2].plot(accuracies)
    axs[2].set_xlabel("epochs")
    axs[2].set_ylabel("accuracy")

    axs[1].plot(grad_norm)
    axs[1].set_xlabel("epochs")
    axs[1].set_ylabel("grad_norm")

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
    X_train = X_train.toarray()
    y_train = y_train - 1.0
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
    X_train = X_train.toarray()
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
    y_train = np.array([y_i % 2 == 0 for y_i in y_train])

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    return X_train, y_train, X_test, y_test


def perform_train_run(dataset, loss_t, method, epochs):

    if dataset == "a9a":
        X_train, y_train, X_test, y_test = download_and_preprocess_a9a()
    elif dataset == "covtype":
        X_train, y_train, X_test, y_test = download_and_preprocess_covtype()
    elif dataset == "ijcnn1":
        X_train, y_train, X_test, y_test = download_and_preprocess_ijcnn1()
    else:
        X_train, y_train, X_test, y_test = download_and_preprocess_mnist()
        print(y_test)

    match method:
        case "gd":
            method = Method.GD
        case "newton":
            method = Method.NEWTON
        case "grn":
            method = Method.GRN
        case "aicn":
            method = Method.AICN
        case "adan":
            method = Method.ADAN
        case "adanp":
            method = Method.ADANP


    print(f"number of samples = {len(y_train)}")
    print("x max/min:", np.max(X_train), np.min(X_train))
    print(type(X_test))
    print("Data sets have been loaded")

    # Training
    if dataset != "mnist":
        y_train = np.clip(y_train, 0.0, 1.0)
        y_test = np.clip(y_test, 0.0, 1.0)

    if loss_t == "ce":
        loss_type = LossFunction.CE
    else:
        loss_type = LossFunction.NCCE
        y_test = y_test * 2 - 1

    lr = MultivarLogReg(method=method, loss_type=loss_type)
    lr.fit(X_train, y_train, epochs=epochs, lr=1, batch_size=None, lbd=0, epsilon=1e-3)
    print("Training complete")

    pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print(f"Test accuracy: {accuracy}")

    return lr, accuracy


if __name__ == '__main__':
    # Argparse 
    parser = argparse.ArgumentParser(description="Select dataset to use.")
    parser.add_argument("dataset", type=str, choices=["a9a", "covtype", "ijcnn1", "mnist"],
                        help="Dataset to use (a9a, covtype, ijcnn1, mnist)")
    parser.add_argument("loss", type=str, choices=["ce", "ncce"], 
                        help="Loss function to use (ce, ncce)")
    parser.add_argument("method", type=str, choices=["gd", "newton", "grn", "aicn", "adan", "adanp"], 
                        help="method to use (gd, newton, m22, cubic, adan)")
    args = parser.parse_args()

    lr, accuracy = perform_train_run(args.dataset, args.loss, args.method, 10)

    print(f"time to convergence = {lr.time_to_convergence}")

    # Plotting
    fig, axs = plt.subplots(1, 3)
    make_plots(lr.losses, lr.train_accuracies, lr.grad_norm, axs, row=0)
    plt.show()
