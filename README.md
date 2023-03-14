This repository contains code to perform image classification using a pre-trained PyTorch model on the ImageNet dataset. The model has been converted to ONNX format for efficient deployment.

Below are the steps to run this code:
1. Download the model weights from this link https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0

2. Create a virtual environment, activate it, then install the requirements.
```
pip install -r requirements.txt
```


3. Convert the dowloaded PyTorch model to an ONNX model.
The file `pytorch_model_weights.onnx` will be created.
```
python convert_to_onnx.py
```

4. Run the tests. The test_onnx.py contains code to test the converted ONNX model on CPU.
There are two sample images in this repository - n01440764_tench.jpeg and n01667114_mud_turtle.jpeg. 
The test_onnx.py script uses these images to verify that the ONNX model outputs the correct class ID and class name.

The model.py contains two classes - OnnxModel and Preprocessor - which provide functionality for loading and predicting with the ONNX model and pre-processing an input image, respectively, will be used in the tests.
```
pytest test_onnx.py
```


5. Deploy the ONNX model.
- Fork the public template on Banana Dev.
- Clone the forked repository to your local machine.
- Copy the contents of this repository to the cloned repository.
- Update the DockerFile to include these file.
```
ADD model.py .
ADD pytorch_model.py .
ADD pytorch_model_weights.onnx .
```
- Create a new app on Banana Dev and link it to your forked repository.
- Push to deploy.

6. Querying the deployed model.
- The test_server.py contains code to make a call to a deployed instance of the model on Banana Dev. It accepts the url of an image and prints the name of the predicted class. Additionally, it reports the time taken to make a call to the deployed model.
- NB: You'll need to use the keys provided in Banana Dev.
