"""Configuration of image analysis module."""

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
             'glevel': 'Img_GreyLevel',           # program will add .tsv or
             'ctrack': 'Img_ContourTracking',     # .json depending on context
             'transform': 'Img_Transform',  # this is to store rotation angle etc.
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
