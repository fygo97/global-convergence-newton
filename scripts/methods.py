from enum import Enum

class Method(Enum):
    GD = 1
    NEWTON = 2
    GRN = 3 # Globally Regularized Newton Algo 1.1 Mishchenko 2023
    AICN = 4 # Affine Invariant Cubinc Newton Algo 1 p. 6 Henzley 2022
    ADAN = 5 # Adaptive Newton Algo 2.1 Mishchenko 2023
    ADANP = 6 # Adaptive Newton Plus Algo 2.3 Mishchenko 2023
    CREG = 7 # 


class LossFunction(Enum):
    CE = 1
    NCCE = 2

class DataSet(Enum):
    A9A = "a9a"
    COVTYPE = "covtype"
    MNIST = "mnist"
    IJCNN1 = "ijcnn1"
