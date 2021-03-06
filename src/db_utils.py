
import psycopg2
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import base64
import numpy as np

model = load_model('../resources/baseline_adam_optimizer.h5')

def get_sql_con():
    user = os.environ['POSTGRES_USER']
    pw = os.environ['POSTGRES_PASSWORD']
    db = os.environ['POSTGRES_DB']
    host = 'db_server'

    con = psycopg2.connect(f'host={host} user={user} password={pw} dbname={db}')
    return con

def predict_class(filename):
    img = prepare_image(filename)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    prediction_string = classes[np.argmax(model.predict(img), axis=-1)[0]]
    prediction_int = np.argmax(model.predict(img), axis=-1)
    return prediction_string, prediction_int

def write_data_to_table(filename):
    con = get_sql_con()

    data = prepare_image(filename)
    _, prediction = predict_class(filename)

    query = f'INSERT INTO image_classes (pixel_data, class) VALUES (%s, %s)'
    cursor = con.cursor()

    cursor.execute(query, (psycopg2.Binary(base64.b64encode(data)), int(prediction)))
    con.commit()
    con.close()
    return

# load and prepare the image
def prepare_image(filename):
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