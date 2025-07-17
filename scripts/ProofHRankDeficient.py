import os
from sklearn.datasets import load_svmlight_file
import numpy as np

def main():
    # Get the absolute path to the script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build full path to 'a9a' inside the scripts folder
    a9a_path = os.path.join(script_dir, "a9a")
    X_a9a, y_a9a = load_svmlight_file(a9a_path)
    X_a9a = X_a9a.toarray()
    rank_a9a = np.linalg.matrix_rank(X_a9a)
    print("Shape:", X_a9a.shape)
    print("Rank:", rank_a9a)
    print("a9a has full rank:", rank_a9a == min(X_a9a.shape))

    # Build full path to 'ijcnn1' inside the scripts folder
    ijcnn1_path = os.path.join(script_dir, "ijcnn1")
    X_ijcnn1, y_ijcnn1 = load_svmlight_file(ijcnn1_path)
    X_ijcnn1 = X_ijcnn1.toarray()
    rank_ijcnn1 = np.linalg.matrix_rank(X_ijcnn1)
    print("Shape:", X_ijcnn1.shape)
    print("Rank:", rank_ijcnn1)
    print("ijcnn1 has full rank:", rank_ijcnn1 == min(X_ijcnn1.shape))

    # Build full path to 'mnist' inside the scripts folder
    mnist_path = os.path.join(script_dir, "mnist.scale")
    X_mnist, y_mnist = load_svmlight_file(mnist_path)
    X_mnist = X_mnist.toarray()
    rank_mnist = np.linalg.matrix_rank(X_mnist)
    print("Shape:", X_mnist.shape)
    print("Rank:", rank_mnist)
    print("mnist has full rank:", rank_mnist == min(X_mnist.shape))

    # Build full path to 'covtype' inside the scripts folder
    covtype_path = os.path.join(script_dir, "covtype.libsvm.binary")
    X_covtype, y_covtype = load_svmlight_file(covtype_path)
    X_covtype = X_covtype.toarray()
    rank_covtype = np.linalg.matrix_rank(X_covtype)
    print("Shape:", X_covtype.shape)
    print("Rank:", rank_covtype)
    print("Full rank:", rank_covtype == min(X_covtype.shape))



if __name__ == "__main__":
    main()
