"""Classes to store transform parameters: rotation, crop, etc.

These parameters are directly applied to images upon loading and are thus
impacting further analysis of the images.
"""

# Standard library
from math import pi
from functools import lru_cache

# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import imgbasics
from imgbasics.cropping import _cropzone_draw
from drapo import linput

# Local imports
from .parameters_base import TransformParameter
from ..viewers import ThresholdSetterViewer


class Grayscale(TransformParameter):
    """Class to store RGB to gray transform.

    grayscale.apply can be True or False
    """

    parameter_type = 'grayscale'

    @property
    def apply(self):
        return self.data.get('apply')

    @apply.setter
    def apply(self, value):
        if value:
            self.img_series.ndim = 2
        else:
            self.img_series.ndim = self.img_series.initial_ndim
        self.data['apply'] = value
        self._update_parameters()


class Rotation(TransformParameter):
    """Class to store and manage rotation angles on series of images."""

    parameter_type = 'rotation'

    def define(self, num=0, vertical=False, **kwargs):
        """Interactively define rotation angle by drawing a line.

        Parameters
        ----------
        - num: image ('num' id) on which to define rotation angle. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - direction: can be horizontal (default, vertical=False) or vertical;
          the drawn line will be brought to this direction after rotation.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data the rotation angle with key "angle"
        """
        self.reset()

        fig, ax = plt.subplots()
        img = self.img_series.read(num=num, rotation=False, crop=False)
        self.img_series._imshow(img, ax=ax, **kwargs)

        direction_name = 'vertical' if vertical else 'horizontal'
        ax.set_title(f'Draw {direction_name} line')

        (x1, y1), (x2, y2) = linput()
        dx = x1 - x2
        dy = y1 - y2
        a, b = (dx, dy) if vertical else (dy, -dx)

        angle = - np.arctan2(a, b) * 180 / pi
        plt.close(fig)

        self.data = {'angle': angle}

    def show(self, num=0, **kwargs):
        """Show the rotated image.

        Parameters
        ----------
        - num: id number of image on which to show the zones (default first one).

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        _, ax = plt.subplots()
        img = self.img_series.read(num=num)
        self.img_series._imshow(img, ax=ax, **kwargs)

        try:
            ax.set_title(f"Rotation: {self.data['angle']:.1f}° (img #{num})")
        except KeyError:
            ax.set_title("No rotation defined")
        return ax

    @property
    def angle(self):
        return self.data.get('angle')

    @angle.setter
    def angle(self, value):
        self.data['angle'] = value
        self._update_parameters()


class Crop(TransformParameter):
    """Class to store and manage global cropping (ROI) on series of images.

    IMPORTANT NOTE: crop zones have to be defined AFTER defining a rotation,
    because they are applied on the coordinates of the rotated image.
    """

    parameter_type = 'crop'

    def define(self, num=0, draggable=False, **kwargs):
        """Interactively define ROI

        Parameters
        ----------
        - num: image ('num' id) on which to define rotation angle. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - draggable: use draggable rectangle from drapo to define crop zones
          instead of clicking to define opposite rectangle corners.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data the (x, y, width, height) as a value in
        a dict with key "zone".
        """
        self.reset()

        img = self.img_series.read(num=num, crop=False)    # rotation is applied here
        default_kwargs = self.img_series._get_imshow_kwargs()
        kwargs = {**default_kwargs, **kwargs}

        _, cropzone = imgbasics.imcrop(img, draggable=draggable, **kwargs)
        self.data = {'zone': cropzone}

    def show(self, num=0, **kwargs):
        """Show the defined ROI on the full image.

        Parameters
        ----------
        - num: id number of image on which to show the zones (default first one).

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        img = self.img_series.read(num=num, crop=False)

        _, ax = plt.subplots()
        self.img_series._imshow(img, ax=ax, **kwargs)

        try:
            _cropzone_draw(ax, self.data['zone'], c='r')
        except KeyError:
            ax.set_title('No crop zone defined')
        else:
            ax.set_title(f'Crop Zone (img #{num})')

        return ax

    @property
    def zone(self):
        return self.data.get('zone')

    @zone.setter
    def zone(self, value):
        self.data['zone'] = value
        self._update_parameters()


class Filter(TransformParameter):
    """Class to store and manage filters (gaussian smoothing, etc.)"""

    parameter_type = 'filter'

    def define(self, num=0, max_size=10, **kwargs):
        """Interactively define filter.

        Parameters
        ----------
        - num: image ('num' id) to display. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data a dict with info about the filter.
        """
        self.reset()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])

        img = self.img_series.read(num=num)
        imshow = self.img_series._imshow(img, ax=ax, **kwargs)

        @lru_cache(maxsize=516)
        def filter_image(size):
            return self.img_series.img_transformer.img_manager.filter(
                img=img,
                filter_type='gaussian',
                size=size,
            )

        def update_image(size):
            self.size = size
            img = filter_image(size)
            imshow.set_array(img)

        self.size = 1

        slider = Slider(ax=ax_slider,
                        label='size',
                        valmin=0,
                        valmax=max_size,
                        valinit=1,
                        valstep=0.1,
                        color='steelblue',
                        alpha=0.5)

        slider.on_changed(update_image)

        self.data = {'type': 'gaussian', 'size': self.size}

        return slider

    @property
    def size(self):
        # auto-select filter type if not specified
        self.type = self.data.get('type', 'gaussian')
        return self.data.get('size')

    @size.setter
    def size(self, value):

        self.data['size'] = value

        # Put default filter type if type is not set yet.
        try:
            self.data['type']
        except KeyError:
            self.data['type'] = 'gaussian'

        self._update_parameters()

    @property
    def type(self):
        return self.data.get('type')

    @type.setter
    def type(self, value):
        self.data['type'] = value
        self._update_parameters()


class Subtraction(TransformParameter):
    """Class to store and manage image subtraction.

    self.reference is an iterable of the images to use for subtraction
    (e.g. self.reference = (0, 1, 2) will use an average of the first images
    as the image to subtract to the current one)

    self.relative adds a division of the reference image to the subtraction
    i.e. I_final = (I - I_ref) / I_ref.
    """

    parameter_type = 'subtraction'

    def _calculate_reference(self, ref_nums):
        imgs = []
        for num in ref_nums:
            imgs.append(self.img_series.read(num=num, subtraction=False))
        img_stack = np.stack(imgs)
        return img_stack.mean(axis=0)

    def _update_reference_image(self):
        self.reference_image = self._calculate_reference(self.reference)

    @property
    def reference(self):
        return self.data.get('reference')

    @reference.setter
    def reference(self, value):
        self.data['reference'] = tuple(value)
        self.reference_image = self._calculate_reference(ref_nums=value)
        self._clear_cache()

    @property
    def relative(self):
        return self.data.get('relative')

    @relative.setter
    def relative(self, value):
        self.data['relative'] = value
        self._update_parameters()


class Threshold(TransformParameter):
    """Class to store and manage image thresholding."""

    parameter_type = 'threshold'

    def define(self, num=0, **kwargs):
        """Interactively define threshold

        Parameters
        ----------
        - num: image ('num' id) on which to define threshold. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        setter = ThresholdSetterViewer(self.img_series, num=num, **kwargs)
        return setter.run()

    @property
    def vmin(self):
        return self.data.get('vmin')

    @vmin.setter
    def vmin(self, value):
        self.data['vmin'] = value
        self._update_parameters()

    @property
    def vmax(self):
        return self.data.get('vmax')

    @vmax.setter
    def vmax(self, value):
        self.data['vmax'] = value
        self._update_parameters()


all_transforms = (
    Grayscale,
    Rotation,
    Crop,
    Filter,
    Subtraction,
    Threshold,
)

Transforms = {
    transform.parameter_type: transform for transform in all_transforms
}
