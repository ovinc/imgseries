"""Class ImgSeries for image series manipulation"""

# Standard library imports
import json

# Nonstandard
import skimage
from skimage import io
from skimage import filters
import imgbasics
from imgbasics.transform import rotate
import numpy as np



class ImageManager:

    pixel_depths = {'uint8': 2**8 - 1,
                    'uint16': 2**16 - 1}

    @staticmethod
    def read(file):
        """load file into image array (file: pathlib Path object)."""
        return io.imread(file)

    @classmethod
    def max_pixel_range(cls, img):
        """Return max pixel value depending on img type, for use in plt.imshow.

        Input
        -----
        img: numpy array

        Output
        ------
        vmin, vmax: max pixel value (None if not float or uint8/16)
        """
        dtype_name = img.dtype.name

        if 'float' in dtype_name:
            return np.nanmin(img), np.nanmax(img)

        return 0, cls.pixel_depths.get(img.dtype.name, None)

    # =========== Define how to transform images (crop, rotate, etc.) ============

    @staticmethod
    def rotate(img, angle):
        """Rotate an image by a given angle"""
        return rotate(img, angle=angle, resize=True, order=3)

    @staticmethod
    def crop(img, zone):
        """Crop an image to zone (X0, Y0, Width, Height)"""
        return imgbasics.imcrop(img, zone)

    @staticmethod
    def subtract(img, img_ref, relative=False):
        """How to subtract a reference image to """
        if not relative:
            return img - img_ref
        else:
            return (img - img_ref) / img_ref

    @classmethod
    def rgb_to_grey(cls, img):
        """How to convert an RGB image to grayscale"""
        _, vmax = cls.max_pixel_range(img)
        img_grey = skimage.color.rgb2gray(img)
        if type(vmax) == int:
            return (img_grey * vmax).astype(img.dtype)
        else:
            return img_grey
        return

    @classmethod
    def filter(cls, img, filter_type='gaussian', size=1):
        """Crop an image to zone (X0, Y0, Width, Height)"""
        _, vmax = cls.max_pixel_range(img)
        if filter_type == 'gaussian':
            img_filtered = filters.gaussian(img, sigma=size)
        if type(vmax) == int:
            return (img_filtered * vmax).astype(img.dtype)
        else:
            return img_filtered


class FileManager:

    @staticmethod
    def from_json(path, filename):
        """"Load json file as a dict.

        path: pathlib object (folder containing the file)
        filename: name of the file without extension
        """
        file = path / (filename + '.json')
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def to_json(data, path, filename):
        """"Save data (dict) to json file.

        data: dictionary of data
        path: pathlib object (folder containing the file)
        filename: name of the file without extension
        """
        file = path / (filename + '.json')
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
