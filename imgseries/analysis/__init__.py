"""Init file for analysis module"""

from .grey_level import GreyLevel
from .contour_tracking import ContourTracking
from .flicker import Flicker
from .front_1d import Front1D


# Define default results classes for analysis ================================
GreyLevelResults = GreyLevel.DefaultResults
ContourTrackingResults = ContourTracking.DefaultResults
Front1DResults = Front1D.DefaultResults
FlickerResults = Flicker.DefaultResults
