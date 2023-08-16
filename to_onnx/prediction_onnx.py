import onnxruntime
import numpy as np
import cv2
from PIL import Image

# plt for debugging on your laptop
import matplotlib.pyplot as plt

from class_label import class_label

# do not import other library


def prediction():
    # write your code
    batch_size = 1
    ch = 3
    rows, cols = 224,224
    model_path = 'C:/Users/Book/Desktop/Work/Re/Lab/to_onnx/model_name.onnx'
    ort_session = onnxruntime.InferenceSession(model_path)

    ######### write your code for read image, normalize (0-1) with max (255)
    ######### and convert to shape (batch_size, ch, h, w)
    dog = Image.open("./dog.jpg").convert("RGB")
    plt.imshow(dog)
    # step 1: dog = normalize
    #print(dog.shape())
    dog = np.array(dog,np.float32)
    dog = ( dog - np.min(dog) ) / ( np.max(dog) -np.min(dog) )
    # step 2: resize
    dog = cv2.resize(dog, dsize=( rows, cols ), interpolation=cv2.INTER_CUBIC)
    # step 3: np.swapaxes
    dog = np.swapaxes( dog , 1 , 0 )
    # step 4: np.swapaxes
    dog = np.swapaxes( dog , 0 , 2 )
    # step 5: reshape
    dog = np.array([dog])

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: dog}
    ort_outs = ort_session.run(None, ort_inputs)

    ######### write your code to get class id and print class label #########
    # Hint: get class label from torch_to_onnx.py
    # Hint: class_id = argmax
    print(np.argmax( ort_outs ) )
    # Hint: class_label[class_id]
    print(class_label[np.argmax( ort_outs )])


if __name__ == "__main__":
    prediction()