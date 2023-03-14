
# test onnx
from load_onnx import load_onnx_model


def test_load_onnx_model_1():
    image_file = "n01667114_mud_turtle.jpeg"
    onnx_file = "pytorch_model_weights.onnx"
    assert load_onnx_model(image_file, onnx_file) == ('n01667114_mud_turtle', 35)
    
    
def test_load_onnx_model_2():
    image_file = "n01440764_tench.jpeg"
    onnx_file = "pytorch_model_weights.onnx"
    assert load_onnx_model(image_file, onnx_file) == ('n01440764_tench', 0)
