import numpy as np

from Framework.Layer import *


class ClassficationLayer(CLayer):
    def __init__(self, classifier):
        self.classifier = classifier

    def forward(self, input, train=True):
        self.input_shape = input.shape[0]
        self.x = input
        self.a = self.classifier.forward(self.x)
        return self.a

    def backward(self, delta_in, flag):
        dZ = delta_in
        return dZ


class CClassifier(object):
    def forward(self, z):
        pass


class Logistic(CClassifier):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(z))
        return a


class Softmax(CClassifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a