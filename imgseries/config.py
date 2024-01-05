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

filenames = {
    'files': 'Img_Files',          # for file info (timing etc.)
    'transform': 'Img_Transform',  # this is to store rotation angle etc.
    'display': 'Img_Display',      # store display options (contrast, cmap etc.)
    'glevel': 'Img_GreyLevel',           # program will add .tsv or
    'ctrack': 'Img_ContourTracking',     # .json depending on context
    'front1d': 'Img_Front1D',
}

# Transform names and the order in which they are applied
image_transforms = (
    'grayscale',
    'rotation',
    'crop',
    'filter',
    'subtraction',
    'threshold',
)

CONFIG = {
    'csv separator': csv_separator,
    'filenames': filenames,
    'image transforms': image_transforms,
    'checked modules': checked_modules,
}
