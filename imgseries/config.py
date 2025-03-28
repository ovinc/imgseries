"""Configuration of image analysis module."""

# Non standard
import skimage

# Additional checked modules to save git/version info
import imgseries
import imgbasics
import filo
import matplotlib
import numpy

CHECKED_MODULES = skimage, imgseries, imgbasics, filo, matplotlib, numpy

CSV_SEPARATOR = '\t'

FILENAMES = {
    'files': 'Img_Files.tsv',          # for file info (timing etc.)
    'transform': 'Img_Transform',  # this is to store rotation angle etc.
    'display': 'Img_Display',      # store display options (contrast, cmap etc.)
}

DEFAULT_CORRECTIONS = (
    'flicker',
    'shaking',
)

DEFAULT_TRANSFORMS = (
    'grayscale',
    'rotation',
    'crop',
    'filter',
    'subtraction',
    'threshold',
)

# How many files can be loaded in the cache
# (files are read only once and stored in memory unless they exceed this limit)
READ_CACHE_SIZE = 1024

# The calculation from loaded data into transformed data can also be cached
TRANSFORM_CACHE_SIZE = 1024

CONFIG = {
    'csv separator': CSV_SEPARATOR,
    'filenames': FILENAMES,
    'checked modules': CHECKED_MODULES,
    'correction order': DEFAULT_CORRECTIONS,
    'transform order': DEFAULT_TRANSFORMS,
    'read cache size': READ_CACHE_SIZE,
    'transform cache size': TRANSFORM_CACHE_SIZE,
}
