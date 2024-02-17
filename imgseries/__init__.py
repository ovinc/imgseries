"""Image analysis tools for image series."""

from .managers import ImageManager, FileManager
from .image_series import ImgSeries, series
from .analysis import GreyLevel, ContourTracking, Front1D, Flicker
from .analysis import GreyLevelResults, ContourTrackingResults
from .analysis import Front1DResults, FlickerResults

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
