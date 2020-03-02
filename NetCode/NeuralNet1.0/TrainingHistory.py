import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#to recode the history of training loss and weights/bias value

class TrainingHistory(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []
        self.weight_history = []
        self.bias_history = []

    def AddLossHistory(self,iteration,loss,weight,bias):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.weight_history.append(weight)
        self.bias_history.append(bias)

    def ShowLossHistory(self,params,xmin=None,xmax=None,ymin=None,ymax=None):
        plt.plot(self.iteration,self.loss_history)
        title = params.toString
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin,xmax,ymin,ymax])
        plt.show()
        return title

    def GetLast(self):
        count = len(self.loss_history)
        return self.loss_history[count-1],self.weight_history[count-1],self.bias_history[count-1]
    #end class