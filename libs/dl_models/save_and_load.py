import config
import onnxruntime
from keras.models import load_model
import onnx
import keras2onnx

class Save_Or_Load_Model:
    def __init__(self):
        pass

    def load_onnx_model(self,path):
        self.onnx_validation(path)
        return onnxruntime.InferenceSession(path)

    def save_onnx_model(self,model,path):
        onnx_model = keras2onnx.convert_keras(model)
        self.onnx_validation(path)
        onnx.save_model(onnx_model, path)

    def load_model(self,path):
        return load_model(path)

    def save_model(self,model,path):
        model.save(path)
        return model

    def onnx_validation(self,path):
        if not path.split('.')[-1]=='onnx':
            raise ValueError('the provided path is not of .onnx format: {}'.format(path))