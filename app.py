from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import configparser


class API:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])

        settings_config = configparser.ConfigParser()
        settings_config.read_file(open(r"config.ini", encoding="utf-8"))
        self.target_size = int(settings_config["DATASET_PARAMS"]["target_size"])

    def prepare_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((self.target_size, self.target_size))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image


    def predict(self):
        model = tf.keras.models.load_model('best_model.h5')
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = self.prepare_image(image)
            prediction = model.predict(processed_image)[0][0]
            class_label = 'Normal' if prediction > 0.5 else 'Cataract'
            confidence = float(prediction) if class_label == 'Normal' else 1 - float(prediction)

            return jsonify({
                'class': class_label,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

