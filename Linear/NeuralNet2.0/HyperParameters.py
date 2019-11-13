from EnumDef import NetType

class HyperParameters(object):
    def __init__(self,input_size,output_size,eta=0.01,max_epoch=1000,batch_size=5,eps=0.1,net_type=NetType.Fitting):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps
        self.net_type = net_type

    def toString(self):
        title = str.format("bz:{0},eta:{1}", self.batch_size, self.eta)
        return title