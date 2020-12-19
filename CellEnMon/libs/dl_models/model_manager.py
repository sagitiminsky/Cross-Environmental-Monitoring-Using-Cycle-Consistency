import CellEnMon.config as config
from keras.models import load_model

class Model_Manager():
    def __init__(self,root,version):
        self.root=root
        self.version=version

    def save(self,model):
        try:
            model.save(self.root+'/'+self.version)
        except Exception as e:
            print("Something went wrong when saving model: {}".format(self.root+'/'+self.version))
            print(e)
            exit(1)

    def load(self):
        try:
            return load_model(self.root+'/'+self.version)
        except Exception as e:
            print("Something went wrong when loading model: {}".format(self.root+'/'+self.version))
            print(e)
            exit(1)