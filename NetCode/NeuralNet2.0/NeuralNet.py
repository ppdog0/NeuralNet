import numpy as np

from TrainingHistory import TrainingHistory
from HyperParameters import HyperParameters
from DataReader import DataReader
from EnumDef import *
from ClassifierFunction import *
from LossFunction import *

class NeuralNet(object):
    def __init__(self,params):
        self.params = params
        self.weight = np.zeros([self.params.input_size, self.params.output_size])
        self.bias = np.zeros([1, self.params.output_size])

    def _forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.weight) + self.bias
        if self.params.net_type == NetType.BinaryClassifier:
            A = Logistic().forward(Z)
            return A
        elif self.params.net_type == NetType.MultipleClassifier:
            A = Softmax().forward(Z)
            return A
        else:
            return Z

    def _backwardBatch(self,batch_x,batch_y,batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0,keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW,dB

    def _update(self,dW,dB):
        self.weight = self.weight - self.params.eta * dW
        self.bias = self.bias - self.params.eta * dB

    def inference(self, x):
        return self._forwardBatch(x)

    def train(self, dataReader, checkpoint=0.1):
        loss_history = TrainingHistory()
        loss_function = LossFunction(self.params.net_type)
        loss = 10
        if self.params.batch_size == -1:
            self.params.batch_size = dataReader.num_train
        max_iteration = (int)(dataReader.num_train/self.params.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)

        for epoch in range(self.params.max_epoch):
            #print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                batch_x,batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                batch_z = self._forwardBatch(batch_x)
                dW,dB = self._backwardBatch(batch_x,batch_y,batch_z)
                self._update(dW,dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.checkLoss(loss_function, dataReader)
                    #print(epoch, iteration, loss, self.weight, self.bias)
                    loss_history.AddLossHistory(epoch * max_iteration + iteration, loss)
                    if loss < self.params.eps:
                        break
                #print("iteration=%d" %iteration)
                #print("weight=" ,self.weight)
                #print("bias=" ,self.bias)
                    #end if
                #end if
            #end for
            if loss < self.params.eps:
                break
        #end for
        loss_history.ShowLossHistory(self.params)
        #print(self.weight,self.bias)

    def checkLoss(self, loss_function, dataReader):
        X, Y = dataReader.GetWholeTrainSamples()
        A = self._forwardBatch(X)
        loss = loss_function.CheckLoss(A, Y)
        return loss