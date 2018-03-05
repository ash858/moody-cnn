import base64
import os
import datetime as dt

from PIL import Image
from flask import Flask
from flask import request
from flask_cors import CORS
from io import BytesIO

from face_detection_model import FaceDetectionModel
from util import Config, ImgUtil

# Register the app
app = Flask(__name__)

# CORS wrapper is needed to use this api from non-web root
CORS(app)

# Setup the paths and config
curr_path = os.path.realpath(__file__)
src_path = os.path.dirname(curr_path)
config_path = f'{src_path}/config.yml'
config = Config(config_path)
model = FaceDetectionModel(config)


@app.route('/stage_image', methods=['POST'])
def stage_image():
    """
    TODO: Document

    :return:
    """
    payload = request.get_json()
    if payload['mood'] and payload['data']:
        mood, data = payload['mood'], payload['data']
        im = _base64_decode_png(data)
        datetime_iso = dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        path = config.face_detection['stage_data_dir']
        filename = f'{path}/{mood}_{datetime_iso}.png'
        im.save(filename, 'PNG')
        return filename
    return 'err'


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if payload['data']:
        data = payload['data']
        im = _base64_decode_png(data)
        vec = ImgUtil.get_image_vector(im, 36, 36)
        return str(model.predict(vec)['classes'])


def _base64_decode_png(base64_enc_str: str) -> Image:
    """
    Parses a base 64 encoded PNG and returns a PIL Image that can be used for saving the image
    or processing it's contents.

    :param base64_enc_str: base 64 encoded representation of the png image
    :return: PIL.Image.Image object
    :exception IOError: If the file cannot be found, or the image cannot be
       opened and identified."""
    return Image.open(BytesIO(base64.b64decode(base64_enc_str)))


if __name__ == '__main__':
    print(config_path)
    print(config)
    m = FaceDetectionModel(config)
    pass
