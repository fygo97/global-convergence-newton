import os
from sklearn.datasets import load_svmlight_file
import numpy as np

def main():
    # Get the absolute path to the script's folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build full path to 'a9a' inside the scripts folder
    file_path = os.path.join(script_dir, "a9a")
    X, y = load_svmlight_file(file_path)
    X = X.toarray()
    rank = np.linalg.matrix_rank(X)
    print("Shape:", X.shape)
    print("Rank:", rank)
    print("Full rank:", rank == min(X.shape))

if __name__ == "__main__":
    main()
