"""General contour measurement and representation"""

# Standard library
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterable

# Misc. package imports
from skimage import measure
import numpy as np
import imgbasics

# Local imports
from .process import rgb_to_grey


# =================== Contour management and calculations ====================


# ------------------------------- Base classes -------------------------------


class ContourCoordinatesBase(ABC):
    """General representation of contour coordinates"""
    pass


class ContourPropertiesBase(ABC):

    @property
    @abstractmethod
    def data(self) -> dict:
        """Dictionary of data that can be saved in json file.

        Does not necessarily have the same keys as the table_columns below
        """
        pass

    @property
    @abstractmethod
    def table_columns(self) -> Iterable[str]:
        """Title of columns if one were to display the properties in a dataframe.

        Combines with self.to_row()
        """
        pass

    @abstractmethod
    def to_table_row(self) -> Iterable:
        """Iterable of data of same length as self.table_columns"""
        pass

    @classmethod
    @abstractmethod
    def from_table_row(cls, row):
        """Create ContourProperties object from row (iterable)"""
        pass


class ContourBase(ABC):

    def __init__(
        self,
        coordinates: ContourCoordinatesBase | None = None,
        properties: ContourPropertiesBase | None = None,
    ):
        """x, y are the coordinates of the contour

        Either coordinates and properties can be None, because contour data
        can store coortinates, properties, or both.
        """
        self.coordinates = coordinates
        # not None when calculate_properties() called
        self.properties = properties

    @abstractmethod
    def _calculate_properties(self) -> ContourPropertiesBase:
        """Calculate properties from coordinates"""
        pass

    def calculate_properties(self) -> None:
        """Calculate centroid, perimeter, area and store it in self.properties"""
        self.properties = self._calculate_properties()

    def reset_properties(self) -> None:
        """Remove calculated properties data."""
        self.properties = None


class ContourFinderBase(ABC):
    """How to extract contours from images and select them following criteria"""

    @abstractmethod
    def find_contours(
        self,
        img: Iterable[float],
        level: float,
    ) -> Iterable[ContourBase]:
        """Define how contours are found on an image given an input level"""
        pass

    @abstractmethod
    def closest_contour_to_click(
        self,
        contours: Iterable[ContourBase],
        click_position: Iterable[float],
    ) -> ContourBase:
        """Define closest contour to position (x, y) for click selection"""
        pass

    @abstractmethod
    def match(
        self,
        contours: Iterable[ContourBase],
        contour_properties: Iterable[ContourPropertiesBase],
    ) -> ContourBase | None:
        """Find closest contour matching reference contour properties

        Return None if no contour found
        """
        pass


# ============================== Usable classes ==============================


@dataclass
class ContourCoordinates(ContourCoordinatesBase):
    """Stores contour property data"""
    x: float
    y: float


class ContourProperties(ContourPropertiesBase):
    """Stores contour property data"""

    table_columns = ('x', 'y', 'p', 'a')

    def __init__(
        self,
        centroid: Iterable[float],
        perimeter: float,
        area: float,
    ):
        self.centroid = centroid
        self.perimeter = perimeter
        self.area = area

    @property
    def data(self) -> dict:
        return vars(self)

    def to_table_row(self) -> Iterable[float]:
        xc, yc = self.centroid
        return (xc, yc, self.perimeter, self.area)

    @classmethod
    def from_table_row(cls, row):
        """Create ContourProperties object from row (iterable)"""
        xc, yc, perimeter, area = row
        return cls(centroid=(xc, yc), perimeter=perimeter, area=area)


class Contour(ContourBase):
    """Class that represents contour data and properties"""

    def _calculate_properties(self):
        """Calculate centroid, perimeter, area and store it in self.properties"""
        ppties = imgbasics.contour_properties(
            x=self.coordinates.x,
            y=self.coordinates.y
        )
        return ContourProperties(**ppties)

    @classmethod
    def from_opencv(cls, contour_data: Iterable[float]):
        x, y = imgbasics.contour_coords(contour_data, source='opencv')
        return cls(coordinates=ContourCoordinates(x=x, y=y))

    @classmethod
    def from_scikit(cls, contour_data: Iterable[float]):
        x, y = imgbasics.contour_coords(contour_data, source='scikit')
        return cls(coordinates=ContourCoordinates(x=x, y=y))

    @classmethod
    def from_hdf5(cls, group):
        pass


class ContourFinder(ContourFinderBase):
    """How to extract contours from images and select them following criteria"""

    def __init__(
        self,
        tolerance_displacement: float | None = None,
        tolerance_area: float | None = None,
    ):
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

    def find_contours(
        self,
        img: Iterable[float],
        level: float,
    ) -> Iterable[Contour]:
        """Define how contours are found on an image."""
        if img.ndim == 2:
            image = img
        else:
            image = rgb_to_grey(img)

        raw_contours = measure.find_contours(image, level)
        contours = [Contour.from_scikit(c) for c in raw_contours]

        return contours

    def closest_contour_to_click(
        self,
        contours: Iterable[Contour],
        click_position: Iterable[float],
    ) -> Contour:
        """Define closest contour to position (x, y) for click selection"""
        raw_contours = [
            (contour.coordinates.x, contour.coordinates.y)
            for contour in contours
        ]
        x, y = imgbasics.closest_contour(raw_contours, click_position, edge=True)
        return Contour(coordinates=ContourCoordinates(x=x, y=y))

    def find_tolerable_contours(
        self,
        contours: Iterable[Contour],
        contour_properties: Iterable[ContourProperties],
    ) -> Iterable[Contour]:
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

    def match(
        self,
        contours: Iterable[Contour],
        contour_properties: Iterable[ContourProperties],
    ) -> Contour | None:
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
