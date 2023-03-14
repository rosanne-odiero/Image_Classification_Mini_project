
# load onnx model
import onnxruntime as onnxrt
from PIL import Image
from pytorch_model import Classifier, BasicBlock
import numpy as np


def load_onnx_model(image_file, onnx_file):
    # load dependency
    obj = Classifier(BasicBlock, [2, 2, 2, 2])

    # create session for the onnx model
    sess = onnxrt.InferenceSession(onnx_file)

    # load and preprocess input
    img = Image.open(image_file)
    inp = obj.preprocess_numpy(img).unsqueeze(0)

    # run session
    out = sess.run(None, {"input": inp.numpy()})
    
    return image_file.split(".")[0], np.argmax(out)
