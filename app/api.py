import datetime as dt
import os

from flask import Flask, escape, request
from flask_cors import CORS

from .img import base64_decode_png

app = Flask(__name__)

# CORS wrapper is needed to use this api from non-web root
CORS(app)

data_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data'


@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'


@app.route('/stage', methods=['POST'])
def stage_img():
    """
    Accepts a base64 encoded png and feature name in a JSON payload
    :return:
    """
    payload = request.get_json()
    if payload['category'] and payload['feature'] and payload['data']:
        category, feature, data = payload['category'].lower(), payload['feature'].lower(), payload['data']
        img = base64_decode_png(data)
        datetime_iso = dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        path = f'{data_path}/raw/{category}/{feature}'
        os.makedirs(path, mode=0o775, exist_ok=True)
        filename = f'{path}/{datetime_iso}.png'
        img.save(filename, 'PNG')
        return filename
    return 'err'
