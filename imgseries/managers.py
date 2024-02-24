"""Image / File managers"""

# Standard library imports
import json

# Nonstandard
import skimage
from skimage import io
from skimage import filters
import imgbasics
from imgbasics.transform import rotate
import numpy as np
import pandas as pd

# local imports
from .config import CONFIG


PIXEL_DEPTHS = {'uint8': 2**8 - 1,
                'uint16': 2**16 - 1}


def max_pixel_range(img):
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
        img_finite = img[np.isfinite(img)]  # remove nan and inf
        return img_finite.min(), img_finite.max()

    elif 'bool' in dtype_name:
        return 0, 1

    return 0, PIXEL_DEPTHS.get(img.dtype.name, None)


class ImageManager:

    @staticmethod
    def read_image(file):
        """load file into image array (file: pathlib Path object)."""
        return io.imread(file)

    @staticmethod
    def read_tiff_stack_slice(file, num):
        """load file into image array (file: pathlib Path object)."""
        return io.imread(file, key=num)

    @staticmethod
    def read_tiff_stack_whole(file):
        """load file into image array (file: pathlib Path object)."""
        return io.imread(file)

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
        """How to subtract a reference image to the image"""
        if not relative:
            return img - img_ref
        else:
            return (img - img_ref) / img_ref

    @staticmethod
    def divide(img, value):
        """Divide image by value, but keep initial data type"""
        # Avoids problems, e.g. np.uint8(257) is actually 1
        temp_img = np.clip(img / value, *max_pixel_range(img))
        return temp_img.astype(img.dtype)

    @staticmethod
    def rgb_to_grey(img):
        """How to convert an RGB image to grayscale"""
        _, vmax = max_pixel_range(img)
        img_grey = skimage.color.rgb2gray(img)
        if type(vmax) is int:
            return (img_grey * vmax).astype(img.dtype)
        else:
            return img_grey

    @staticmethod
    def filter(img, filter_type='gaussian', size=1):
        """Crop an image to zone (X0, Y0, Width, Height)"""
        _, vmax = max_pixel_range(img)
        if filter_type == 'gaussian':
            img_filtered = filters.gaussian(img, sigma=size)
        else:
            raise ValueError(f'{filter_type} filter not implemented')
        if type(vmax) is int:
            return (img_filtered * vmax).astype(img.dtype)
        else:
            return img_filtered

    @staticmethod
    def threshold(img, vmin=None, vmax=None):
        """Threshold image (vmin <= v <= vmax --> 1, else 0).

        Return binary (boolean) image (True / False).
        """
        if None in (vmin, vmax):
            val_min, val_max = max_pixel_range(img)
            vmin = val_min if vmin is None else vmin
            vmax = val_max if vmax is None else vmax
        condition = (img >= vmin) & (img <= vmax)
        return np.where(condition, True, False)


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

    @staticmethod
    def from_tsv(path, filename):
        """"Load tsv data file as a dataframe.

        path: pathlib object (folder containing the file)
        filename: name of the file without extension
        """
        file = path / (filename + '.tsv')
        return pd.read_csv(file, index_col='num', sep=CONFIG['csv separator'])

    @staticmethod
    def to_tsv(data, path, filename):
        """"Save dataframe to tsv data file.

        data: pandas dataframe
        path: pathlib object (folder containing the file)
        filename: name of the file without extension
        """
        file = path / (filename + '.tsv')
        data.to_csv(file, sep=CONFIG['csv separator'])
