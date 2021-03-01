
from flask import Flask
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model('../resources/test_model.h5')

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route('/')
def hello_world():
    img = load_image('../resources/sample_image.png')
    result = model.predict_classes(img)

    return f'{result}'
