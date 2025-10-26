"""General contour measurement and representation"""

# Standard library
from dataclasses import dataclass

# Misc. package imports
from skimage import measure
import numpy as np
import imgbasics

# Local imports
from .process import rgb_to_grey


# =================== Contour management and calculations ====================


@dataclass
class ContourCoordinates:
    """Stores contour property data"""

    x: float
    y: float

    @property
    def data(self):
        return vars(self)


@dataclass
class ContourProperties:
    """Stores contour property data"""

    centroid: tuple
    perimeter: float
    area: float

    @property
    def data(self):
        return vars(self)


class Contour:
    """Class that represents contour data and properties"""

    def __init__(self, coordinates=None, properties=None):
        """x, y are the coordinates of the contour

        Either coordinates and properties can be None, because contour data
        can store coortinates, properties, or both.
        """
        self.coordinates = coordinates
        # not None when calculate_properties() called
        self.properties = properties

    def calculate_properties(self):
        """Calculate centroid, perimeter, area and store it in self.properties"""
        ppties = imgbasics.contour_properties(
            x=self.coordinates.x,
            y=self.coordinates.y
        )
        self.properties = ContourProperties(**ppties)

    def reset_properties(self):
        """Remove calculated properties data."""
        self.properties = None

    @classmethod
    def from_opencv(cls, contour_data):
        x, y = imgbasics.contour_coords(contour_data, source='opencv')
        return cls(coordinates=ContourCoordinates(x=x, y=y))

    @classmethod
    def from_scikit(cls, contour_data):
        x, y = imgbasics.contour_coords(contour_data, source='scikit')
        return cls(coordinates=ContourCoordinates(x=x, y=y))


# ======================= Contour calculation methods ========================


class ContourCalculator:
    """How to extract contours from images and select them following criteria"""

    def __init__(self, tolerance_displacement=None, tolerance_area=None):
        """Init contour calculator object

        Parameters
        ----------

        tolerance_displacement : float
            if None (default), no restriction on displacements
            if value = d > 0, do not consider displacements more than d pixels

        tolerance_area : float
            if None (default), no restriction on area variations of contours
            if value = x > 0, do not consider relative variation in area of
            more than x.
        """
        # Tolerance in displacement and areas to match contours
        self.tolerance_displacement = tolerance_displacement
        self.tolerance_area = tolerance_area

    def find_contours(self, img, level):
        """Define how contours are found on an image."""
        if img.ndim == 2:
            image = img
        else:
            image = rgb_to_grey(img)

        raw_contours = measure.find_contours(image, level)
        contours = [Contour.from_scikit(c) for c in raw_contours]

        return contours

    def closest_contour_to_click(self, contours, click_position):
        """Define closest contour to position (x, y) for click selection"""
        raw_contours = [
            (contour.coordinates.x, contour.coordinates.y)
            for contour in contours
        ]
        x, y = imgbasics.closest_contour(raw_contours, click_position, edge=True)
        return Contour(coordinates=ContourCoordinates(x=x, y=y))

    def find_tolerable_contours(self, contours, contour_properties):
        """Find all contours that match tolerance criteria for matching given contour"""

        ok_contours = []

        for contour in contours:

            if self.tolerance_displacement is not None:
                x1, y1 = contour_properties.centroid
                x2, y2 = contour.properties.centroid
                d = np.hypot(x2 - x1, y2 - y1)
                if d > self.tolerance_displacement:
                    continue

            if self.tolerance_area is not None:
                a = abs(contour.properties.area)
                a0 = abs(contour_properties.area)
                x = abs((a - a0)) / a0
                if x > self.tolerance_area:
                    continue

            ok_contours.append(contour)

        return ok_contours

    def match(self, contours, contour_properties):
        """Find closest contour matching reference contour properties

        tolerance_displacement: max displacement in px
        tolerance_area: max relative change in area
        """
        for contour in contours:
            if contour.properties is None:
                contour.calculate_properties()

        tolerable_contours = self.find_tolerable_contours(
            contours,
            contour_properties,
        )

        # No contours found --> return None
        if len(tolerable_contours) < 1:
            return

        # Among all tolerated contours, return that which is closest (centroid)
        displacements = []
        for contour in tolerable_contours:
            x1, y1 = contour_properties.centroid
            x2, y2 = contour.properties.centroid
            d = np.hypot(x2 - x1, y2 - y1)
            displacements.append(d)

        imin = displacements.index(min(displacements))
        return tolerable_contours[imin]
