class CLayer(object):
    def __init__(self, layer_type):
        self.layer_type = layer_type

    def initialize(self, folder):
        pass

    def train(self, input, train=True):
        pass

    def update(self):
        pass

    def save_parameters(self, folder, name):
        pass

    def load_parameters(self, folder, name):
        pass