import numpy as np

from EnumDef import NetType

class LossFunction(object):
    def __init__(self,net_type):
        self.net_type = net_type

    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)

        return loss

    def MSE(self, A, Y, count):
        p1 = A - Y
        LOSS = np.multiply(p1,p1)
        loss = LOSS.sum()/count/2
        return loss

    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)

        p4 = np.multiply(p1, p2)
        p5 = np.multiply(Y, p3)

        LOSS = np.sum(-(p4+p5))
        loss = LOSS / count
        return loss

    def CE3(self, A, Y, count):
        p1 = np.log(A)
        p2 = np.multiply(Y, p1)
        LOSS = np.sum(-p2)
        loss = LOSS / count
        return loss