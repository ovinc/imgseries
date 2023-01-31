"""Init file for analysis module"""

from .grey_level import GreyLevel
from .contour_tracking import ContourTracking


# Define default results classes for analysis ================================
GreyLevelResults = GreyLevel.DefaultResults
ContourTrackingResults = ContourTracking.DefaultResults
