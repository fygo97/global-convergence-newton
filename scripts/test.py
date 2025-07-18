import numpy as np
from logreg import MultivarLogReg
from scipy.special import expit

def predict(x, weights):
    '''
        Returns a class label prediction
    '''
    logits = x @ weights        # shape: (n_samples, num_classes)
    probs = expit(logits)
    prediction = np.array([1 if p > 0.5 else 0 for p in probs])
    return prediction


if __name__ == '__main__':
    x = np.random.randn(100, 100) / 25
    weights = np.ones(100)
    print(x)
    print(predict(x, weights))




