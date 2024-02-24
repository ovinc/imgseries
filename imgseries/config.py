"""Configuration of image analysis module."""

# Non standard
import skimage

# Additional checked modules to save git/version info
import imgseries
import imgbasics
import filo
import matplotlib
import numpy

checked_modules = skimage, imgseries, imgbasics, filo, matplotlib, numpy

csv_separator = '\t'

FILENAMES = {
    'files': 'Img_Files',          # for file info (timing etc.)
    'transform': 'Img_Transform',  # this is to store rotation angle etc.
    'display': 'Img_Display',      # store display options (contrast, cmap etc.)
}

# Correction names and the order in which they are applied
IMAGE_CORRECTIONS = (
    'flicker',
    'shaking',
)

# Transform names and the order in which they are applied
IMAGE_TRANSFORMS = (
    'grayscale',
    'rotation',
    'crop',
    'filter',
    'subtraction',
    'threshold',
)

CONFIG = {
    'csv separator': csv_separator,
    'filenames': FILENAMES,
    'image transforms': IMAGE_TRANSFORMS,
    'image corrections': IMAGE_CORRECTIONS,
    'checked modules': checked_modules,
}
