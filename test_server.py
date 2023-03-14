
import banana_dev as banana
import time


def test_server():
    api_key = "99729d08-8b8f-4c25-ac54-c300768a3fd8"
    model_key = "ff287d6a-6c5f-427b-bf29-800d116e345c"
    model_inputs = {'input': 'https://raw.githubusercontent.com/MTailorEng/mtailor_mlops_assessment/main/n01440764_tench.jpeg'} # anything you want to send to your model

    start = time.time()
    out = banana.run(api_key, model_key, model_inputs)
    stop = time.time()
    
    return out, stop-start
