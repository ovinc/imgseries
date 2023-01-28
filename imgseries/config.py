"""Configuration of image analysis module."""

# Standard library
import json

# Non standard
import skimage
from imgbasics.transform import rotate

# Additional checked modules to save git/version info
import imgseries
import imgbasics
import filo
import matplotlib
import numpy

checked_modules = skimage, imgseries, imgbasics, filo, matplotlib, numpy


# ================================== Config ==================================


csv_separator = '\t'

filenames = {'files': 'Img_Files',  # for file info (timing etc.)
             'transform': 'Img_Transform',  # this is to store rotation angle etc.
             'display': 'Img_Display',  # store display options (contrast, cmap etc.)
             'glevel': 'Img_GreyLevel',           # program will add .tsv or
             'ctrack': 'Img_ContourTracking',     # .json depending on context
             }


# ======================== Define how to load images =========================


def _read(file):
    """load file into image array (file: pathlib Path object)."""
    return skimage.io.imread(file)


def _rgb_to_grey(img):
    """How to convert an RGB image to grayscale"""
    return skimage.color.rgb2gray(img)

# =========== Define how to transform images (crop, rotate, etc.) ============

def _rotate(img, angle):
    """Rotate an image by a given angle"""
    return rotate(img, angle=angle, resize=True, order=3)

def _crop(img, zone):
    """Crop an image to zone (X0, Y0, Width, Height)"""
    return imgbasics.imcrop(img, zone)

# =========================== Pixel and bit depths ===========================

def _max_possible_pixel_value(img):
    """Return max pixel value depending on img type, for use in plt.imshow.

    Input
    -----
    img: numpy array

    Output
    ------
    vmax: max pixel value (int or float or None)
    """
    if img.dtype == 'uint8':
        return 2**8 - 1
    elif img.dtype == 'uint16':
        return 2**16 - 1
    else:
        return None

# ================================= File I/O =================================

def _from_json(path, filename):
    """"Load json file as a dict.

    path: pathlib object (folder containing the file)
    filename: name of the file without extension
    """
    file = path / (filename + '.json')
    with open(file, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

def _to_json(data, path, filename):
    """"Save data (dict) to json file.

    data: dictionary of data
    path: pathlib object (folder containing the file)
    filename: name of the file without extension
    """
    file = path / (filename + '.json')
    with open(file, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
