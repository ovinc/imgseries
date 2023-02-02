"""Image analysis tools for image series."""

from .general import ImageManager, FileManager
from .image_series import ImgSeries, series
from .analysis import GreyLevel, ContourTracking
from .analysis import GreyLevelResults, ContourTrackingResults
from .analysis.front_1d import Front1D

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
