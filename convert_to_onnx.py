
# convert to onnx
import torch
from PIL import Image
from pytorch_model import Classifier, BasicBlock


def convert_to_onnx():
    # load the pretrained model
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load("pytorch_model_weights.pth"))
    model.eval()

    # define an input tensor
    img = Image.open("./n01667114_mud_turtle.JPEG")
    inp = model.preprocess_numpy(img).unsqueeze(0) 

    # export to onnx
    torch.onnx.export(model, inp, "pytorch_model_weights.onnx")

if __name__ == "__main__":
    convert_to_onnx()
