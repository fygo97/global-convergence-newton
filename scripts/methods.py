from enum import Enum

class Method(Enum):
    GD = 1
    NEWTON = 2
    M22 = 3
    CUBIC = 4


class LossFunction(Enum):
    CE = 1
    NCCE = 2

class DataSet(Enum):
    A9A = "a9a"
    COVTYPE = "covtype"
    MNIST = "mnist"
    IJCNN1 = "ijcnn1"
