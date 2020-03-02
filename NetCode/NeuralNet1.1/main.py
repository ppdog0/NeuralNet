import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HyperParameters import HyperParameters
from DataReader import DataReader
from NeuralNet import NeuralNet

file_name = "./ch05.npz"

if __name__ == "__main__":
    reader = DataReader(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()

    hp = HyperParameters(2,1,eta=0.01,max_epoch=200,batch_size=10,eps = 1e-5)
    net = NeuralNet(hp)
    net.train(reader,checkpoint=0.1)
    print("W=",net.weight)
    print("B=",net.bias)

    x1 = 15
    x2 = 93
    x = np.array([x1,x2]).reshape(1,2)
    x_new = reader.NormalizePredicateData(x)
    z = net.inference(x_new)
    print("Z=",z)
    Z_true = z*reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z_true=",Z_true)