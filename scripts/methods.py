from enum import Enum

class Method(Enum):
    GD = 1
    NEWTON = 2


class LossFunction(Enum):
    CE = 1
    NCCE = 2
