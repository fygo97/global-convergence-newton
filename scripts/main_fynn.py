import numpy as np
import matplotlib.pyplot as plt
from loss_fynn import CE, NCCE
from solvers import Solvers
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import os

# ---------- SETTINGS ----------
#ALL_DATASETS = ["a9a", "covtype", "ijcnn1", "mnist"]
ALL_DATASETS = ["a9a"]
lambda_ = 0.001
alpha = 1.0
epochs = 2        # You can change this
batch_size = 3000  # You can change this
test_size = 0.25   # You can change this

# ---------- PATH SETUP ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- DATA LOADER ----------
def load_data(name):
    if name == "a9a":
        Xtr, ytr = load_svmlight_file(os.path.join(SCRIPT_DIR, "a9a"))
        Xte, yte = load_svmlight_file(os.path.join(SCRIPT_DIR, "a9a.t"), n_features=Xtr.shape[1])
    elif name == "covtype":
        Xtr, ytr = load_svmlight_file(os.path.join(SCRIPT_DIR, "covtype.libsvm.binary"))
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=test_size)
    elif name == "ijcnn1":
        Xtr, ytr = load_svmlight_file(os.path.join(SCRIPT_DIR, "ijcnn1"))
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=test_size)
    else:
        Xtr, ytr = load_svmlight_file(os.path.join(SCRIPT_DIR, "mnist.scale"))
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=test_size)

    return Xtr.toarray().astype(np.float32), ytr.astype(np.float32), Xte.toarray().astype(np.float32), yte.astype(np.float32)

# ---------- BATCHED MISHCHENKO ----------
def batched_mishchenko(X, y, loss_class, weights_0, epochs, batch_size, lambda_=None, alpha=None):
    weights = weights_0.copy()
    n_samples = X.shape[0]
    iterates = [weights.copy()]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # Instantiate correct loss
            if loss_class == CE:
                batch_loss = CE(X_batch, y_batch)
            else:
                batch_loss = NCCE(X_batch, y_batch, lambda_, alpha)

            batch_solver = Solvers(batch_loss.value, batch_loss.grad, batch_loss.hessian)
            weights = batch_solver.mishchenko(weights, max_iter=1,tol=1e-6)

        iterates.append(weights.copy())
    return iterates

# ---------- CE LOOP ----------
for dataset in ALL_DATASETS:
    X_train, y_train, X_test, y_test = load_data(dataset)
    y_train = np.clip(y_train, 0, 1)
    y_test = np.clip(y_test, 0, 1)

    weights_0 = np.random.randn(X_train.shape[1]) * 0.01

    iterates = batched_mishchenko(X_train, y_train, CE, weights_0, epochs, batch_size)

    losses, grad_norms, test_accs = [], [], []
    L1_full = CE(X_train, y_train)
    for w in iterates:
        losses.append(L1_full.value(w))
        grad_norms.append(np.linalg.norm(L1_full.grad(w)))
        logits = X_test @ w
        preds = (logits >= 0).astype(np.float32)
        test_accs.append(np.mean(preds == y_test))

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title(f"{dataset.upper()} CE (L1) Loss")
    plt.subplot(1, 3, 2)
    plt.plot(test_accs)
    plt.title(f"{dataset.upper()} CE (L1) Test Accuracy")
    plt.subplot(1, 3, 3)
    plt.plot(grad_norms)
    plt.title(f"{dataset.upper()} CE (L1) Grad Norm")
    plt.tight_layout()
    plt.savefig(f"{dataset}_ce_results.png")
    plt.close()

# ---------- NCCE LOOP ----------
for dataset in ALL_DATASETS:
    X_train, y_train, X_test, y_test = load_data(dataset)
    y_train = 2 * np.clip(y_train, 0, 1) - 1
    y_test = 2 * np.clip(y_test, 0, 1) - 1

    weights_0 = np.random.randn(X_train.shape[1]) * 0.01

    iterates = batched_mishchenko(X_train, y_train, NCCE, weights_0, epochs, batch_size, lambda_, alpha)

    losses, grad_norms, test_accs = [], [], []
    L2_full = NCCE(X_train, y_train, lambda_, alpha)
    for w in iterates:
        losses.append(L2_full.value(w))
        grad_norms.append(np.linalg.norm(L2_full.grad(w)))
        logits = X_test @ w
        preds = np.where(logits >= 0, 1, -1)
        test_accs.append(np.mean(preds == y_test))

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title(f"{dataset.upper()} NCCE (L2) Loss")
    plt.subplot(1, 3, 2)
    plt.plot(test_accs)
    plt.title(f"{dataset.upper()} NCCE (L2) Test Accuracy")
    plt.subplot(1, 3, 3)
    plt.plot(grad_norms)
    plt.title(f"{dataset.upper()} NCCE (L2) Grad Norm")
    plt.tight_layout()
    plt.savefig(f"{dataset}_ncce_results.png")
    plt.close()
