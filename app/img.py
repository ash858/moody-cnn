import base64
import os
import shutil
from io import BytesIO

import numpy as np
from PIL import Image

from .config import data_path


def open_image(path: str) -> Image:
    return Image.open(path)


def get_batch_image_vectors(path, width, height):
    return np.array([get_image_vector(open_image(f'{path}/{file}'), width, height)
                     for file in os.listdir(path) if file.endswith('.png')])


def get_image_vector(img: Image, width, height) -> np.ndarray:
    resized_img = resize(greyscale(img), width, height)
    # Normalize the vector so that the values are between 0 and 1
    return np.array(resized_img, dtype=np.float32) / 255


def greyscale(image: Image) -> Image:
    return image.convert('L')


def resize(image: Image, width: int, height: int) -> Image:
    return image.resize((width, height), Image.BILINEAR)


def copy(src: str, dest: str) -> None:
    shutil.copyfile(src, dest)


def base64_decode_png(base64_enc_str: str) -> Image:
    """
    Parses a base 64 encoded PNG and returns a PIL Image that can be used for saving the image
    or processing it's contents.

    :param base64_enc_str: base 64 encoded representation of the png image
    :return: PIL.Image.Image object
    :exception IOError: If the file cannot be found, or the image cannot be
       opened and identified."""
    return Image.open(BytesIO(base64.b64decode(base64_enc_str)))


if __name__ == "__main__":
    res = get_batch_image_vectors(f"{data_path}/raw/mood/sad", 64, 64)
    print(res.shape)
