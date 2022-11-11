import base64
import io
from io import StringIO
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image

model = load_model("vgg19.h5")


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)
    # Process the image frame
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    size = (224,224)
    frame = tf.image.resize(frame,size)
    x = tf.keras.preprocessing.image.img_to_array(frame)
    x = np.expand_dims(x, axis= 0)
    pred = model.predict(x)
    pred = np.argmax(pred)
    if pred == 1:
        p = "no fire"
        
    else:
        p = "fire"
    # emit the frame back
    emit('response_back',p)

if __name__ == '__main__':
    socketio.run(app,host="127.0.0.1")