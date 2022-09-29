"""Image analysis tools for image series."""

from .general import ImgSeries
from .grey_level import GreyLevel
from .contour_tracking import ContourTracking
from .imbibitionfront_tracking import ImbibitionTracking
from .plots import plot_contour_evolution

from importlib_metadata import version

__author__ = 'Olivier Vincent'
__version__ = version("imgseries")
