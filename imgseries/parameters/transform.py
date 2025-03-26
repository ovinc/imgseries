"""Classes to store transform parameters: rotation, crop, etc.

These parameters are directly applied to images upon loading and are thus
impacting further analysis of the images.
"""

# Standard library
from functools import lru_cache

# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import imgbasics
import imgbasics.transform
import imgbasics.cropping
from drapo import linput

# Local imports
from .parameters_base import TransformParameter
from ..viewers import ThresholdSetterViewer
from ..process import rgb_to_grey, double_threshold, gaussian_filter


class Grayscale(TransformParameter):
    """Class to store RGB to gray transform.

    grayscale.active can be True or False
    """

    name = 'grayscale'

    @property
    def active(self):
        return self.data.get('active')

    @active.setter
    def active(self, value):
        self.data['active'] = value
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        return rgb_to_grey(img)


class Rotation(TransformParameter):
    """Class to store and manage rotation on series of images."""

    name = 'rotation'

    def define(self, num=0, vertical=False, **kwargs):
        """Interactively define rotation angle by drawing a line.

        Parameters
        ----------
        num : int
            image ('num' id) on which to define rotation angle. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        vertical : bool
            the direction can be horizontal (default, vertical=False)
            or vertical;
            the drawn line will be brought to this direction after rotation.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data the rotation angle with key "angle"
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

        angle = - np.arctan2(a, b) * 180 / np.pi
        plt.close(fig)

        self.angle = angle

    def show(self, num=0, **kwargs):
        """Show the rotated image.

        Parameters
        ----------
        num : int
            id number of image on which to show the zones (default first one).

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
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
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        return imgbasics.transform.rotate(
            img,
            angle=self.angle,
            resize=True,
            order=3,
        )


class Crop(TransformParameter):
    """Class to store and manage global cropping (ROI) on series of images.

    IMPORTANT NOTE: crop zones have to be defined AFTER defining a rotation,
    because they are applied on the coordinates of the rotated image.
    """

    name = 'crop'

    def define(self, num=0, draggable=False, **kwargs):
        """Interactively define ROI

        Parameters
        ----------
        num : int
            image ('num' id) on which to define rotation angle. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        draggable : bool
            If True, use draggable rectangle from drapo to define crop zones
            instead of clicking to define opposite rectangle corners.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data the (x, y, width, height) as a value in
            a dict with key "zone".
        """
        self.reset()

        img = self.img_series.read(num=num, crop=False)    # rotation is applied here
        default_kwargs = self.img_series._get_imshow_kwargs()
        kwargs = {**default_kwargs, **kwargs}

        _, cropzone = imgbasics.imcrop(img, draggable=draggable, **kwargs)
        self.zone = cropzone

    def show(self, num=0, **kwargs):
        """Show the defined ROI on the full image.

        Parameters
        ----------
        num : int
            id number of image on which to show the zones (default first one).

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        img = self.img_series.read(num=num, crop=False)

        _, ax = plt.subplots()
        self.img_series._imshow(img, ax=ax, **kwargs)

        try:
            imgbasics.cropping._cropzone_draw(ax, self.data['zone'], c='r')
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
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        return imgbasics.imcrop(img, self.zone)


class Filter(TransformParameter):
    """Class to store and manage filters (gaussian smoothing, etc.)"""

    name = 'filter'

    def define(self, num=0, max_size=10, **kwargs):
        """Interactively define filter.

        Parameters
        ----------
        num : int
            image ('num' id) to display. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        max_size : float
            max filter size settable with the slider

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        None
            but stores in self.data a dict with info about the filter.
        """
        self.reset()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])

        img = self.img_series.read(num=num)
        imshow = self.img_series._imshow(img, ax=ax, **kwargs)

        @lru_cache(maxsize=516)
        def filter_image(size):
            return gaussian_filter(img, size=size)

        def update_image(size):
            self.size = size
            img = filter_image(size)
            imshow.set_array(img)

        self.size = 1

        slider = Slider(
            ax=ax_slider,
            label='size',
            valmin=0,
            valmax=max_size,
            valinit=1,
            valstep=0.1,
            color='steelblue',
            alpha=0.5,
        )

        slider.on_changed(update_image)

        self.data = {'type': 'gaussian', 'size': self.size}

        return slider

    @property
    def size(self):
        return self.data.get('size')

    @size.setter
    def size(self, value):
        self.data['size'] = value
        self._update_others()

    @property
    def type(self):
        return self.data.get('type')

    @type.setter
    def type(self, value):
        self.data['type'] = value
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        return gaussian_filter(img, size=self.size)


class Subtraction(TransformParameter):
    """Class to store and manage image subtraction.

    self.reference is an iterable of the images to use for subtraction
    (e.g. self.reference = (0, 1, 2) will use an average of the first images
    as the image to subtract to the current one)

    self.relative adds a division of the reference image to the subtraction
    i.e. I_final = (I - I_ref) / I_ref.
    """

    name = 'subtraction'

    def _calculate_reference(self, ref_nums):
        imgs = []
        for num in ref_nums:
            imgs.append(self.img_series.read(num=num, subtraction=False))
        img_stack = np.stack(imgs)
        return img_stack.mean(axis=0)

    def _update_reference_image(self):
        self.reference_image = self._calculate_reference(self.reference)

    def _update_parameter(self):
        if not self.is_empty:
            self._update_reference_image()

    @property
    def reference(self):
        return self.data.get('reference')

    @reference.setter
    def reference(self, value):
        self.data['reference'] = tuple(value)
        self._update_reference_image()
        self._update_others()

    @property
    def relative(self):
        return self.data.get('relative')

    @relative.setter
    def relative(self, value):
        self.data['relative'] = value
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        if not self.relative:
            return img - self.reference_image
        else:
            return img / self.reference_image - 1


class Threshold(TransformParameter):
    """Class to store and manage image thresholding."""

    name = 'threshold'

    def define(self, num=0, **kwargs):
        """Interactively define threshold

        Parameters
        ----------
        num : int
            image ('num' id) on which to define threshold. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
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
        self._update_others()

    @property
    def vmax(self):
        return self.data.get('vmax')

    @vmax.setter
    def vmax(self, value):
        self.data['vmax'] = value
        self._update_others()

    def apply(self, img):
        """How to apply the transform on an image array

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        return double_threshold(img, vmin=self.vmin, vmax=self.vmax)


All_Transforms = (
    Grayscale,
    Rotation,
    Crop,
    Filter,
    Subtraction,
    Threshold,
)

TRANSFORMS = {
    Transform.name: Transform for Transform in All_Transforms
}
