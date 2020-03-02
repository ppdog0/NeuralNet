import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from HyperParameters import HyperParameters
from DataReader import DataReader
from TrainingHistory import TrainingHistory

class NeuralNet(object):
    def __init__(self,params):
        self.params = params
        self.weight = np.zeros([self.params.input_size,self.params.output_size])
        self.bias = np.zeros([1,self.params.output_size])

    def _forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.weight) + self.bias
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

        if self.params.batch_size == -1:
            self.params.batch_size = dataReader.num_train
        max_iteration = (int)(dataReader.num_train/self.params.batch_size)
        checkpoint_iteration = (int)(max_iteration*checkpoint)

        for epoch in range(self.params.max_epoch):
            print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                batch_x,batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                batch_z = self._forwardBatch(batch_x)
                dW,dB = self._backwardBatch(batch_x,batch_y,batch_z)
                self._update(dW,dB)

                total_iteration = epoch*max_iteration + iteration
                if (total_iteration+1)%checkpoint_iteration == 0:
                    loss = self._checkLoss(dataReader)
                    print(epoch,iteration,loss)
                    loss_history.AddLossHistory(epoch*max_iteration+iteration, loss, self.weight[0,0], self.bias[0,0])
                    if loss <self.params.eps:
                        break
                    #end if
                #end if
            #end for
            if loss < self.params.eps:
                break
        #end for
        loss_history.ShowLossHistory(self.params)
        print(self.weight,self.bias)

        #self.loss_contour(dataReader,loss_history,self.params.batch_size,epoch*max_iteration+iteration)

    def _checkLoss(self,dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self._forwardBatch(X)
        LOSS = (Z-Y)**2
        loss = LOSS.sum()/m/2
        return loss

    def loss_contour(self,dataReader,loss_history,batch_size,iteration):
        last_loss, result_w, result_b = loss_history.GetLast()
        len1 = 2
        len2 = 1
        w = np.linspace(result_w-1,result_w+1,len1)
        b = np.linspace(result_b-1,result_b+1,len2)
        W, B = np.meshgrid(w,b)
        lens = len1*len2
        X, Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = np.dot(X,W.ravel().reshape(2,1)) + B.ravel().reshape(1,2)
        Loss1 = (Z-Y)**2
        Loss2 = Loss1.sum(axis=0,keepdims=True)/m
        Loss3 = Loss2.reshape(len1,len2)
        plt.contour(W,B,Loss3,levels=np.logspace(-5,5,100),norm=LogNorm(),cmap=plt.cm.jet)

        #show w,b trace
        w_history = loss_history.weight_history
        b_history = loss_history.bias_history
        plt.plot(w_history,b_history)
        plt.xlabel("w")
        plt.ylabel("b")
        title = str.format("bacthsize={0},iteration={1},eta={2},w={3:.3f},b={4:.3f}",batch_size,iteration,self.params.eta,result_w,result_b)
        plt.title(title)

        plt.axis([result_w-1,result_w+1,result_b-1,result_b+1])
        plt.show()