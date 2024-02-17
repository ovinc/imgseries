"""Classes to store parameters specific to analyses: contours, ROIs etc."""

# Standard library
from functools import lru_cache

# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import imgbasics
from imgbasics.cropping import _cropzone_draw

# Local imports
from .parameters_base import AnalysisParameter
from ..managers import max_pixel_range


class Zones(AnalysisParameter):
    """Class to store and manage areas of interest on series of images."""

    parameter_type = 'zones'

    def define(self, n=1, num=0, draggable=False, **kwargs):
        """Interactively define n zones in image.

        Parameters
        ----------
        - n: number of zones to analyze (default 1)

        - num: image ('num' id) on which to select crop zones. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - draggable: use draggable rectangle from drapo to define crop zones
          instead of clicking to define opposite rectangle corners.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data a dict with every cropzone used during
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

            _, cropzone = imgbasics.imcrop(img,
                                           color=clr,
                                           message=msg,
                                           draggable=draggable,
                                           ax=ax,
                                           closefig=False,
                                           **kwargs)

            name = f'zone {k + 1}'
            zones[name] = cropzone

        plt.close(fig)

        self.data = zones

    def show(self, num=0, **kwargs):
        """show the defined zones on image (image id num if specified)

        Parameters
        ----------
        - num: id number of image on which to show the zones (default first one).

        - kwargs: any keyword-argument to pass to imshow() (overrides default
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


class Contours(AnalysisParameter):
    """Class to store and manage reference contours param in image series."""

    parameter_type = 'contours'

    def define(self, n=1, num=0, **kwargs):
        """Interactively define n contours on an image at level level.

        Parameters
        ----------
        - level: grey level at which to define threshold to detect contours

        - n: number of contours

        - num: image identifier (num=0 corresponds to first image in first folder)

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data a dictionary with keys:
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
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        # Interactively select contours of interest on image -----------------

        positions = {}

        for k in range(1, n + 1):

            ax.set_title(f'Contour {k} / {n}')
            fig.canvas.draw()
            fig.canvas.flush_events()

            pt, = plt.ginput()

            contour = imgbasics.closest_contour(contours, pt, edge=True)
            x, y = imgbasics.contour_coords(contour, source='scikit')

            ax.plot(x, y, linewidth=1, c='y')
            plt.pause(0.01)

            xc, yc = imgbasics.contour_properties(x, y)['centroid']

            name = f'contour {k}'
            positions[name] = (xc, yc)  # store position of centroid

        plt.close(fig)

        self.data = {'position': positions,
                     'level': level,
                     'image': num}

    def show(self, **kwargs):
        """Show reference contours used for contour tracking.

        Parameters
        ----------
        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        num = self.data['image']
        level = self.data['level']
        positions = self.data['position']

        # Load image, crop it, and calculate contours
        img = self.analysis.img_series.read(num)
        contours = self.analysis._find_contours(img, level)

        _, ax = plt.subplots()
        self.analysis.img_series._imshow(img, ax=ax, **kwargs)

        # Find contours closest to reference positions and plot them
        for contour in contours:
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=1, c='b')

        # Interactively select contours of interest on image -----------------
        for pt in positions.values():
            contour = imgbasics.closest_contour(contours, pt, edge=False)
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        ax.set_title(f'img #{num}, grey level {level}')

        plt.show()

        return ax

    def load(self, filename=None):
        """Redefined here because threshold data is contained in contours data'
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        self.data = all_data[self.parameter_type]
        self.analysis.threshold.value = self.data['level']


class Threshold(AnalysisParameter):
    """Class to store and manage grey level thresholds (e.g. to define contours.)"""

    parameter_type = 'threshold'

    def define(self, num=0, **kwargs):
        """Interactively define threshold

        Parameters
        ----------
        - num: image ('num' id) to display. Note that this number can be
               different from the name written in the image filename, because
               num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data a dict with threshold value
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
                x, y = imgbasics.contour_coords(contour, source='scikit')
                line, = ax.plot(x, y, linewidth=2, c='r')
                self.lines.append(line)

        level_min, level_max = max_pixel_range(img)
        level_step = 1 if type(level_max) == int else None

        level_start = level_max // 2
        draw_contours(level_start)

        slider = Slider(ax=ax_slider,
                        label='level',
                        valmin=level_min,
                        valmax=level_max,
                        valinit=level_start,
                        valstep=level_step,
                        color='steelblue',
                        alpha=0.5)

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
            self.data = all_data[self.parameter_type]
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
