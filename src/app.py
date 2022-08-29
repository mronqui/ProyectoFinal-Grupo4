import os
import cv2

from flask import Flask, request, render_template, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import transform_image, process_ocr, extract_bruto_from_text, extract_neto_from_text, extract_tara_from_text

app = Flask(__name__)
model = load_model('../models/remito_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    files = request.files.getlist("files")
    for file in files:
        #filename = file.filename
        try:
            # leo el byte y lo convierto a una imagen
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            # hago el resize de la imagen
            img_resized = cv2.resize(img,(160,160))
            img_resized = tf.convert_to_tensor(img_resized[:,:,:3])
            img_resized = tf.expand_dims(img_resized, 0)
            output = model.predict(img_resized)
            output = tf.nn.sigmoid(output)
            if output[0] < 0.5:
                transformed = transform_image(img)
                text = process_ocr(transformed)
                predictions.append({'bruto': extract_bruto_from_text(text), 'tara': extract_tara_from_text(text), 'neto': extract_neto_from_text(text)})
            else:
                predictions.append({'bruto': 'Not found', 'tara': 'Not found', 'neto': 'Not found'})
        except Exception as e:
                print('Error:', e)
    
    response = app.response_class(
        response=json.dumps(predictions),
        mimetype='application/json'
    )
    return response

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)