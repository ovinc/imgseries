"""Image analysis tools for image series."""

from .general import ImgSeries, series
from .analysis.grey_level import GreyLevel, GreyLevelResults
from .analysis.contour_tracking import ContourTracking, ContourTrackingResults
from .analysis.front_1d import Front1D

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
