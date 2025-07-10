import numpy as np
import requests
import urllib3
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import os
from Logreg import CustomLogReg
import matplotlib.pyplot as plt
from methods import Method


def make_plots(losses, axs, row = 0):
    xs = range(len(losses))
    axs[row, 0].plot(xs, losses)
    axs[row, 0].set_xlabel("epochs")
    axs[row, 0].set_ylabel("loss")
    axs[row, 1].plot(lr.train_accuracies)
    axs[row, 0].set_xlabel("epochs")
    axs[row, 0].set_ylabel("accuracy")

    

if __name__ == '__main__':

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
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    print("x max/min:", np.max(X_train), np.min(X_train))
    print(type(X_test))
    print("Data sets have been loaded")

    # Prepocessing
    y_train = np.clip(y_train, 0.0, 1.0)
    y_test = np.clip(y_test, 0.0, 1.0)

    # Train model
    epochs = 10
    lr = CustomLogReg(Method.NEWTON)
    lr.fit(X_train, y_train, epochs=epochs, lr=0.001)
    print("Training complete")
    pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    # Newton's method
    lr_newton = CustomLogReg(Method.NEWTON)
    # lr_newton.fit(X_train, y_train, epochs=epochs)


    #plotting
    fig, axs = plt.subplots(2, 2)
    make_plots(lr.losses, axs)
    plt.show()
