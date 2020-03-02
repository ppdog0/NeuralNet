import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HyperParameters import HyperParameters
from DataReader import DataReader
from NeuralNet import NeuralNet

file_name = "./housing.data"

def draw(reader,net):
    plt.plot(reader.XTrain,reader.YTrain)
    plt.show()

if __name__ == "__main__":
    reader = DataReader()
    reader.ReadData()

    reader.NormalizeX()
    reader.NormalizeY()

    hp = HyperParameters(13,1,eta=0.001,max_epoch=2000,batch_size=50,eps = 1e-5)
    net = NeuralNet(hp)
    net.train(reader,checkpoint=0.2)
    print("W=",net.weight)
    print("B=",net.bias)

'''
    x1=0.00632
    x2=18.00
    x3=2.310
    x4=0
    x5=0.5380
    x6=6.5750
    x7=65.20
    x8=4.0900
    x9=1
    x10=296.0
    x11=15.30
    x12=396.90
    x13=4.98

    x=np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13]).reshape(1,13)
    x_new=reader.NormalizePredicateData(x)
    z=net.inference(x_new)
    z_true=z*reader.Y_norm[0,1] + reader.Y_norm[0,0]
'''
    #print("z=",z_true)