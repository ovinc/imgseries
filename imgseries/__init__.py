"""Image analysis tools for image series."""

from .series import ImgSeries, series, ImgStack, stack
from .analysis import GreyLevel, ContourTracking, Front1D, Flicker
from .analysis import GreyLevelResults, ContourTrackingResults
from .analysis import Front1DResults, FlickerResults

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
