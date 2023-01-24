"""Image analysis tools for image series."""

from .general import ImgSeries
from .grey_level import GreyLevel
from .contour_tracking import ContourTracking
from .front_1d import Front1D

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
