import os
import yaml
from PIL import Image
import shutil
import hashlib
import numpy as np


class Config:
    """
    Simple config container to pass into the models. Reads yaml files.
    """

    def __init__(self, config_file_path: str):
        """
        Opens the config path and parses the yaml file
        :param config_file_path:
        """
        with open(config_file_path) as f:
            self.data = yaml.safe_load(f)

    def __getattr__(self, item) -> str:
        return self.data[item]

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self) -> str:
        return str(self.data)


class ImgUtil:
    def __init__(self):
        pass

    @classmethod
    def get_batch_image_vectors(cls, path, width, height):
        return np.array([cls.get_image_vector(cls.open_image(f'{path}/{file}'), width, height)
                for file in os.listdir(path) if file.endswith('.png')])

    @classmethod
    def open_image(cls, path: str) -> Image:
        return Image.open(path)

    @classmethod
    def get_image_vector(cls, img: Image, width, height) -> np.ndarray:
        resized_img = cls.resize(cls.greyscale(img), width, height)
        # Normalize the vector so that the values are between 0 and 1
        return np.array(resized_img, dtype=np.float32) / 255

    @classmethod
    def greyscale(cls, image: Image) -> Image:
        return image.convert('L')

    @classmethod
    def resize(cls, image: Image, width: int, height: int) -> Image:
        return image.resize((width, height), Image.BILINEAR)

    @classmethod
    def move(cls, src: str, dest: str) -> None:
        shutil.copyfile(src, dest)


if __name__ == '__main__':
    conf = Config(os.path.realpath('src/config.yml'))
    sample_img = conf['face_detection']['train_data_dir'] + '/sad/20.png'
    print(sample_img)
    img = ImgUtil.open_image(sample_img)
    print(ImgUtil.get_batch_image_vectors(conf['face_detection']['train_data_dir'] + '/sad', 36, 36).shape)
