"""Configuration of image analysis module."""

import skimage

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
             'ctrack': 'Img_ContourTracking',
             'itrack': 'Img_ContourImbibitionTracking'}     # .json depending on context


# ======================== Define how to load images =========================


def _read(file):
    """load file into image array (file: pathlib Path object)."""
    return skimage.io.imread(file)
