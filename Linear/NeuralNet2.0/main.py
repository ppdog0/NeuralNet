import numpy as np
import matplotlib.pyplot as plt

from HyperParameters import HyperParameters
from DataReader import DataReader
from NeuralNet import NeuralNet
from EnumDef import NetType

def ShowData(X,Y):
    for i in range(X.shape[0]):
        if Y[i] == 0:
            plt.plot(X[i,0],X[i,1], '.', c='r')
        elif Y[i] == 1:
            plt.plot(X[i,0],X[i,1], 'x', c='g')
        elif Y[i] == 2:
            plt.plot(X[i,0],X[i,1], '^', c='b')
    plt.show()

if __name__ == "__main__":
    num_input = 4
    num_category = 3

    reader = DataReader()
    reader.ReadData()

    ShowData(reader.XRaw,reader.YRaw)

    reader.NormalizeX()
    reader.ToOneHot(num_category,0)

    hp = HyperParameters(num_input,num_category,eta=0.1,max_epoch=2000,batch_size=10,eps = 1e-5,net_type=NetType.MultipleClassifier)
    net = NeuralNet(hp)
    net.train(reader,checkpoint=1)
    print("W=",net.weight)
    print("B=",net.bias)