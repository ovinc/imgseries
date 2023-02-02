"""Classes to store various parameters: crop zones, rotations, contours, etc."""

# Standard library
from math import pi
from functools import lru_cache

# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import imgbasics
from imgbasics.cropping import _cropzone_draw
from drapo import linput


# =============================== Base classes ===============================


class ImageParameter:
    """Base class to define common methods for different parameters."""

    parameter_type = None  # define in subclasses (e.g. "zones")

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. ImgSeries)
        """
        self.img_series = img_series  # ImgSeries object on which to define zones
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def __repr__(self):
        return f'{self.__class__.__name__} object {self.data}'

    def load(self, filename=None):
        """Load parameter data from .json file and put it in self.data.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        self.data = all_data[self.parameter_type]

    def reset(self):
        """Reset parameter data (e.g. rotation angle zero, ROI = total image, etc.)"""
        self.data = {}

    @property
    def is_empty(self):
        return not self.data


class DisplayParameter(ImageParameter):
    """Base class for global dispaly options (contrast changes, colormaps, etc.)

    These parameters DO NOT impact analysis (only options for image display)
    """

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        pass


class TransformParameter(ImageParameter):
    """Base class for global transorms on image series (rotation, crop etc.)

    These parameters DO impact analysis and are stored in metadata.
    """

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.img_series.load_transform(filename=filename)

    def reset(self):
        """Reset parameter data (e.g. rotation angle zero, ROI = total image, etc.)"""
        self.data = {}
        self._update_parameters()

    def _clear_cache(self):
        """If images are stored in a cache, clear it so that the new transform
        parameter can be taken into account upon read()"""
        if self.img_series.cache:
            self.img_series.read.cache_clear()

    def _update_parameters(self):
        """What to do when a parameter is updated"""
        self._clear_cache()

        subtraction = self.img_series.subtraction
        if not subtraction.is_empty:
            subtraction._update_reference_image()

        grayscale = self.img_series.grayscale
        if grayscale.is_empty or not grayscale.apply:
            self.img_series.ndim = self.img_series.initial_ndim
        else:
            self.img_series.ndim = 2


class AnalysisParameter(ImageParameter):
    """Base class for parameters used in analysis (contours, zones, etc.)"""

    def __init__(self, analysis):
        """Init parameter object.

        Parameters
        ----------
        - analysis: object of an analysis class (e.g. GreyLevel)
        """
        self.analysis = analysis  # ImgSeries object on which to define zones
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.analysis.results._load_metadata(filename=filename)


# =============== Display parameters (do not impact analysis) ================


class Contrast(DisplayParameter):
    """Class to store and manage contrast / brightness change."""

    parameter_type = 'contrast'

    def define(self, num=0, **kwargs):
        """Interactively define brightness / contrast

        Parameters
        ----------
        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data the (x, y, width, height) as a value in
        a dict with key "zone".
        """
        fig = plt.figure(figsize=(12, 5))
        ax_img = fig.add_axes([0.05, 0.05, 0.5, 0.9])

        img = self.img_series.read(num=num)
        imshow = self.img_series._imshow(img, ax=ax_img, **kwargs)

        img_object, = ax_img.get_images()
        vmin, vmax = img_object.get_clim()
        vmin_min, vmax_max = self.img_series.image_manager.max_pixel_range(img)
        v_step = 1 if type(vmax_max) == int else None

        img_flat = img.flatten()
        vmin_img = img_flat.min()
        vmax_img = img_flat.max()

        ax_hist = fig.add_axes([0.7, 0.45, 0.25, 0.5])
        ax_hist.hist(img_flat, bins='auto')
        ax_hist.set_xlim((vmin_min, vmax_max))

        min_line = ax_hist.axvline(vmin, color='k')
        max_line = ax_hist.axvline(vmax, color='k')

        ax_slider_min = fig.add_axes([0.7, 0.2, 0.25, 0.04])
        ax_slider_max = fig.add_axes([0.7, 0.28, 0.25, 0.04])

        slider_min = Slider(ax=ax_slider_min,
                            label='min',
                            valmin=vmin_min,
                            valmax=vmax_max,
                            valinit=vmin,
                            valstep=v_step,
                            color='steelblue',
                            alpha=0.5)

        slider_max = Slider(ax=ax_slider_max,
                            label='max',
                            valmin=vmin_min,
                            valmax=vmax_max,
                            valinit=vmax,
                            valstep=v_step,
                            color='steelblue',
                            alpha=0.5)

        ax_btn_reset = fig.add_axes([0.7, 0.05, 0.07, 0.06])
        btn_reset = Button(ax_btn_reset, 'Full')

        ax_btn_auto = fig.add_axes([0.79, 0.05, 0.07, 0.06])
        btn_auto = Button(ax_btn_auto, 'Auto')

        ax_btn_ok = fig.add_axes([0.88, 0.05, 0.07, 0.06])
        btn_ok = Button(ax_btn_ok, 'OK')

        def update_min(value):
            imshow.norm.vmin = value
            min_line.set_xdata((value, value))

        def update_max(value):
            imshow.norm.vmax = value
            max_line.set_xdata((value, value))

        def reset_contrast(event):
            slider_min.set_val(vmin_min)
            slider_max.set_val(vmax_max)

        def auto_contrast(event):
            slider_min.set_val(vmin_img)
            slider_max.set_val(vmax_img)

        def validate(event):
            self.data = {'vmin': slider_min.val, 'vmax': slider_max.val}
            plt.close(fig)

        slider_min.on_changed(update_min)
        slider_max.on_changed(update_max)

        btn_reset.on_clicked(reset_contrast)
        btn_auto.on_clicked(auto_contrast)
        btn_ok.on_clicked(validate)

        return slider_min, slider_max, btn_reset, btn_auto, btn_ok

    @property
    def vmin(self):
        try:
            return self.data['vmin']
        except KeyError:
            return

    @vmin.setter
    def vmin(self, value):
        self.data['vmin'] = value

    @property
    def vmax(self):
        try:
            return self.data['vmax']
        except KeyError:
            return

    @vmax.setter
    def vmax(self, value):
        self.data['vmax'] = value

    @property
    def limits(self):
        return self.vmin, self.vmax

    @vmax.setter
    def vmax(self, value):
        vmin, vmax = value
        self.vmin = vmin
        self.vmax = vmax


class Colors(DisplayParameter):
    """Class to store and manage colormaps used for display"""

    parameter_type = 'colors'

    def define(self, num=0, **kwargs):
        """Interactively define colormap

        Parameters
        ----------
        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)

        Output
        ------
        None, but stores in self.data the (x, y, width, height) as a value in
        a dict with key "zone".
        """
        fig = plt.figure(figsize=(8, 5))
        ax_img = fig.add_axes([0.05, 0.05, 0.7, 0.9])

        img = self.img_series.read(num=num)
        imshow = self.img_series._imshow(img, ax=ax_img, **kwargs)

        img_object, = ax_img.get_images()
        initial_cmap = img_object.get_cmap().name

        ax_btns = {}
        btns = {}

        x = 0.8
        y = 0.15
        w = 0.12
        h = 0.08
        pad = 0.05

        base_color = 'whitesmoke'
        select_color = 'lightblue'

        def change_colormap(name):

            def on_click(event):

                imshow.set_cmap(name)
                self.cmap = name

                btns[name].color = select_color

                for btn_name, btn in btns.items():
                    if btn_name != name:
                        btn.color = base_color

                fig.canvas.draw()

            return on_click

        for cmap in 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis':

            ax = fig.add_axes([x, y, w, h])

            if cmap == initial_cmap:
                color = select_color
            else:
                color = base_color

            btn = Button(ax, cmap, color=color, hovercolor=select_color)
            btn.on_clicked(change_colormap(name=cmap))

            ax_btns[cmap] = ax
            btns[cmap] = btn

            y += h + pad

        return btns

    @property
    def cmap(self):
        try:
            return self.data['cmap']
        except KeyError:
            return

    @cmap.setter
    def cmap(self, value):
        self.data['cmap'] = value


# ============================= Basic transforms =============================


class Grayscale(TransformParameter):
    """Class to store RGB to gray transform.

    grayscale.apply can be True or False
    """

    parameter_type = 'grayscale'

    @property
    def apply(self):
        try:
            return self.data['apply']
        except KeyError:
            return

    @apply.setter
    def apply(self, value):
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
        self.img_series.rotation.reset()

        fig, ax = plt.subplots()
        img = self.img_series.read(num=num, transform=False)
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
        try:
            return self.data['angle']
        except KeyError:
            return

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
        self.img_series.crop.reset()

        img = self.img_series.read(num=num)    # rotation is applied here
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
        img = self.img_series.read(num=num, transform=False)

        if not self.img_series.rotation.is_empty:
            img = self.img_series._rotate(img)

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
        try:
            return self.data['zone']
        except KeyError:
            return

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
        self.img_series.filter.reset()

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])

        img = self.img_series.read(num=num)
        imshow = self.img_series._imshow(img, ax=ax, **kwargs)

        @lru_cache(maxsize=516)
        def filter_image(size):
            return self.img_series.image_manager.filter(img,
                                                        filter_type='gaussian',
                                                        size=size)

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
        try:
            return self.data['size']
        except KeyError:
            return

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
        try:
            return self.data['type']
        except KeyError:
            return

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
        try:
            return self.data['reference']
        except KeyError:
            return

    @reference.setter
    def reference(self, value):
        self.data['reference'] = tuple(value)
        self.reference_image = self._calculate_reference(ref_nums=value)
        self._clear_cache()

    @property
    def relative(self):
        try:
            return self.data['relative']
        except KeyError:
            return

    @relative.setter
    def relative(self, value):
        self.data['relative'] = value
        self._update_parameters()


# ================== Parameters for specific analysis types ==================


class Zones(AnalysisParameter):
    """Class to store and manage areas of interest on series of images."""

    parameter_type = 'zones'

    def define(self, n=1, num=0, draggable=False, **kwargs):
        """Interactively define n zones image.

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

        image_manager = self.analysis.img_series.image_manager

        level_min, level_max = image_manager.max_pixel_range(img)
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
