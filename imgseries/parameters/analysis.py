"""Classes to store parameters specific to analyses: contours, ROIs etc."""

# Standard library
from dataclasses import dataclass
from functools import lru_cache

# Non-standard modules
import drapo
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import imgbasics
from imgbasics.cropping import _cropzone_draw

# Local imports
from .parameters_base import AnalysisParameter
from ..process import max_pixel_range


# ============================ Zones of interest =============================


class Zones(AnalysisParameter):
    """Class to store and manage areas of interest on series of images."""

    name = 'zones'

    def define(self, n=1, num=0, draggable=False, **kwargs):
        """Interactively define n zones in image.

        Parameters
        ----------
        n : int
            number of zones to analyze (default 1)

        num : int
            image ('num' id) on which to select crop zones. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        draggable : bool
            use draggable rectangle from drapo to define crop zones
            instead of clicking to define opposite rectangle corners.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data a dict with every cropzone used during
            the analysis, with:
            Keys: 'zone 1', 'zone 2', etc.
            Values: tuples (x, y, width, height)
        """
        fig, ax = plt.subplots()

        img = self.analysis.img_series.read(num=num)
        self.analysis.img_series._imshow(img, ax=ax, **kwargs)

        zones = {}

        for k in range(n):

            msg = f'Select zone {k + 1} / {n}'

            # line not drawn, just used to set default color and legend
            line, = ax.plot(1, 1, linestyle=None, label=f'zone {k + 1}')
            clr = line.get_color()

            _, cropzone = imgbasics.imcrop(
                img,
                color=clr,
                message=msg,
                draggable=draggable,
                ax=ax,
                closefig=False,
                **kwargs,
            )

            name = f'zone {k + 1}'
            zones[name] = cropzone

        plt.close(fig)

        self.data = zones

    def show(self, num=0, **kwargs):
        """show the defined zones on image (image id num if specified)

        Parameters
        ----------
        num : int
            id number of image on which to show the zones (default first one).

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        img = self.analysis.img_series.read(num=num)

        fig, ax = plt.subplots()
        self.analysis.img_series._imshow(img, ax=ax, **kwargs)

        ax.set_title(f'Analysis Zones (img #{num})')

        for k, zone in enumerate(self.data.values()):
            # line not drawn, just used to set default color and legend
            line, = ax.plot(1, 1, linestyle=None, label=f'zone {k + 1}')
            clr = line.get_color()
            _cropzone_draw(ax, zone, c=clr)

        ax.legend()
        fig.tight_layout()

        return ax


# ================================= Contours =================================


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


class Contours(AnalysisParameter):
    """Class to store and manage reference contours param in image series."""

    name = 'contours'

    def define(self, n=1, num=0, **kwargs):
        """Interactively define n contours on an image at level level.

        Parameters
        ----------
        level : int or float
            grey level at which to define threshold to detect contours

        n : int
            number of contours

        num : int
            image identifier (num=0 corresponds to first image in first folder)

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data a dictionary with keys:
            'position', 'level', 'image'
        """
        level = self.analysis.threshold.value

        fig, ax = plt.subplots()

        img = self.analysis.img_series.read(num=num)
        contours = self.analysis._find_contours(img, level)

        # Plot all contours found --------------------------------------------

        self.analysis.img_series._imshow(img, ax=ax, **kwargs)
        ax.set_xlabel('Left click on vicinity of contour to select.')

        for contour in contours:
            ax.plot(contour.coordinates.x, contour.coordinates.y, linewidth=2, c='r')

        # Interactively select contours of interest on image -----------------

        properties = {}

        for k in range(n):

            ax.set_title(f'Contour {k + 1} / {n}')
            fig.canvas.draw()
            fig.canvas.flush_events()

            clickpt, = drapo.ginput()

            contour = self.analysis._closest_contour_to_click(
                contours,
                clickpt,
            )

            ax.plot(contour.coordinates.x, contour.coordinates.y, linewidth=1, c='y')
            plt.pause(0.01)

            name = f'contour {k + 1}'

            contour.calculate_properties()
            properties[name] = contour.properties.data

        plt.close(fig)

        self.data = {
            'properties': properties,
            'level': level,
            'image': num,
        }

    @property
    def properties(self):
        """Generate contour properties objects based on dicts"""
        ppties = {}
        for name, ppty in self.data['properties'].items():
            ppties[name] = ContourProperties(**ppty)
        return ppties

    def show(self, **kwargs):
        """Show reference contours used for contour tracking.

        Parameters
        ----------
        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        num = self.data['image']
        level = self.data['level']
        all_properties = self.data['properties']

        # Load image, crop it, and calculate contours ------------------------
        img = self.analysis.img_series.read(num)
        contours = self.analysis._find_contours(img, level)

        _, ax = plt.subplots()
        self.analysis.img_series._imshow(img, ax=ax, **kwargs)

        # Find contours closest to reference positions and plot them ---------
        for contour in contours:
            ax.plot(contour.coordinates.x, contour.coordinates.y, linewidth=1, c='b')

        # Interactively select contours of interest on image -----------------
        for properties in all_properties.values():
            contour_properties = ContourProperties(**properties)
            contour = self.analysis._match(contours, contour_properties)
            ax.plot(contour.coordinates.x, contour.coordinates.y, linewidth=2, c='r')

        ax.set_title(f'img #{num}, grey level {level}')

        plt.show()

        return ax

    def load(self, filename=None):
        """Redefined here because threshold data is contained in contours data'
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        self.data = all_data[self.name]
        self.analysis.threshold.value = self.data['level']


# ================================ Threshold =================================


class Threshold(AnalysisParameter):
    """Class to store and manage grey level thresholds (e.g. to define contours.)"""

    name = 'threshold'

    def define(self, num=0, **kwargs):
        """Interactively define threshold

        Parameters
        ----------
        num : int
            image ('num' id) to display. Note that this number can be
            different from the name written in the image filename, because
            num always starts at 0 in the first folder.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data a dict with threshold value
            (key 'value') and accessible by self.value
        """
        self.reset()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])

        img = self.analysis.img_series.read(num=num)
        self.analysis.img_series._imshow(img, ax=ax, **kwargs)

        @lru_cache(maxsize=516)
        def calculate_contours(level):
            return self.analysis._find_contours(img, level)

        self.lines = []

        def draw_contours(level):
            contours = calculate_contours(level)
            for contour in contours:
                line, = ax.plot(contour.coordinates.x, contour.coordinates.y, linewidth=2, c='r')
                self.lines.append(line)

        level_min, level_max = max_pixel_range(img)
        level_step = 1 if type(level_max) is int else None

        level_start = level_max // 2
        draw_contours(level_start)

        slider = Slider(
            ax=ax_slider,
            label='level',
            valmin=level_min,
            valmax=level_max,
            valinit=level_start,
            valstep=level_step,
            color='steelblue',
            alpha=0.5,
        )

        self.data = {'value': level_start}

        def update_level(level):
            self.data['value'] = level
            for line in self.lines:
                line.remove()
            self.lines = []
            draw_contours(level=level)

        slider.on_changed(update_level)

        return slider

    def load(self, filename=None):
        """Redefined here in because some old analyses have this value stored
        in 'contours' and not 'threshold'
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        try:
            self.data = all_data[self.name]
        except KeyError:
            self.value = all_data['contours']['level']

    @property
    def value(self):
        try:
            return self.data['value']
        except KeyError:
            return

    @value.setter
    def value(self, val):
        self.data['value'] = val
