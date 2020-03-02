import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = None
        self.YTrain = None
        self.XRaw = None
        self.YRaw = None

    #read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XRaw = data['data']
            self.YRaw = data['label']
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
            self.num_train = self.XTrain.shape[0]
        else:
            raise Exception("Cannot find train file!!")
        #end if

    #get single training data
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return  x, y

    #get batch traning data
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP

    #data normalization
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape)
        num_feature = self.XRaw.shape[1]
        self.X_norm = np.zeros([num_feature,2])
        for i in range(num_feature):
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)

            self.X_norm[i,0] = min_value

            self.X_norm[i,1] = max_value - min_value
            new_col = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
            X_new[:,0] = new_col
        #end for
        self.XTrain = X_new

    #normalize data by self range and min_value
    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,0]
        return X_new

    def NormalizeY(self):
        self.Y_norm = np.zeros([1,2])
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)

        self.Y_norm[0,0] = min_value
        self.Y_norm[0,1] = max_value - min_value

        Y_new = (self.YRaw - self.Y_norm[0,0]) / self.Y_norm[0,1]
        self.YTrain = Y_new