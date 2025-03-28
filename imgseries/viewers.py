"""Create live plots for analysis on image sequences"""

# Standard library
from abc import abstractmethod

# Nonstandard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from filo import DataViewerBase, FormattedAnalysisViewerBase

# Local imports
from .process import max_pixel_range, double_threshold


class ImgSeriesViewer(DataViewerBase):
    """Matplotlib viewer to inspect image series (no analysis).

    See DataViewerBase for details.
    """

    def __init__(self, img_series):
        """Init Image series viewer

        Parameters
        ----------

        img_series : ImgSeries or ImgStack

        transform : bool
            if True (default), apply global rotation and crop (if defined)
            if False, use raw images.

        **kwargs
            any keyword-argument to pass to imshow().
        """
        super().__init__()
        self.img_series = img_series

    # ============ Method definitions required by DataViewerBase =============

    def _create_figure(self):
        self.fig, self.ax = plt.subplots()
        self.axs = self.ax,

    def _initialize(self):
        """Anything else to do before first plots"""
        pass

    def _get_data(self, num):
        img = self.img_series.read(num, transform=self.transform)
        return {'num': num, 'image': img}

    def _first_plot(self, data):
        self.imshow = self.img_series._imshow(
            data['image'],
            ax=self.ax,
            transform=self.transform,
            **self.kwargs,
        )
        self._display_info(data)
        self.ax.axis('off')
        self.updated_artists = [self.imshow]

    def _update_plot(self, data):
        self.imshow.set_array(data['image'])
        self._display_info(data)

    # ========================= Other useful methods =========================

    def _display_info(self, data):
        num = data['num']
        if self.img_series.is_stack:
            title = 'Image'
        else:
            title = self.img_series.files[num].name
        raw_info = ' [RAW]' if not self.transform else ''
        self.ax.set_title(f'{title} (#{num}){raw_info}')


class AnalysisViewer(FormattedAnalysisViewerBase):
    """Matplotlib viewers to display analysis results alongside images.

    (Base class, to subclass)
    """

    # ========================= Misc. useful methods =========================

    @staticmethod
    def _autoscale(ax):
        ax.relim()  # without this, axes limits change don't work
        ax.autoscale(axis='both')

    def _create_image(self, data):
        """Necessitates that self.ax_img is defined by self._create_figure()"""
        self.ax_img.set_title(f"img #{data['num']}")
        self.imshow = self.analysis.img_series._imshow(
            data["image"],
            ax=self.ax_img,
            **self.kwargs,
        )

    def _update_image(self, data):
        self.ax_img.set_title(f"img #{data['num']}")
        self.imshow.set_array(data['image'])

    def _get_color(self):
        """This is a hack to get automatic colors.

        Necessitates that self.ax_analysis is defined by _create_image()
        """
        noline, = self.ax_analysis.plot([], [])
        return noline.get_color()

    # ========================= Methods to subclass ==========================

    @abstractmethod
    def _create_figure(self):
        """Create figure and axes

        Has to define at least self.fig., and self.axs if self.axs is not
        defined in self._first_plot()
        """
        pass

    @abstractmethod
    def _first_plot(self, data):
        """What to do the first time data arrives on the _plot.

        self.updated_artists must be defined here.
        self.axs must be defined here, except if done in self._create_figure()

        Parameters
        ----------
        data : Any
            Data generated by the analysis

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _update_plot(self, data):
        """What to do upon iterations of the _plot after the first time.

        Parameters
        ----------
        data : Any
            Data generated by the analysis

        Returns
        -------
        None
        """
        pass


# ==================== Interactive setting of parameters =====================


class DoubleSliderBase:
    """Viewer with double slider to set contrast/threshold"""

    def __init__(self, img_series, num=0, **kwargs):
        """Interactively define brightness / contrast or threshold

        Parameters
        ----------
        img_series: ImgSeries object or equivalent (e.g. ImgStack).

        num : int
            image ('num' id) on which to define contrast. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        self.img_series = img_series
        self.num = num
        self.kwargs = kwargs
        self.parameter = None  # parameter that slider acts upon (define in subclass)

    def _update_min(self, value):
        """What to do when min slider is updated. Define in subclass"""
        pass

    def _update_max(self, value):
        """What to do when max slider is updated. Define in subclass"""
        pass

    def _set_contrast(self, vmin, vmax):
        """Set contrast/threshold limits to a specified range (min, max)"""
        self.widgets['slider min'].set_val(vmin)
        self.widgets['slider max'].set_val(vmax)

    def _reset_contrast(self, event):
        return self._set_contrast(*self.init_range)

    def _full_contrast(self, event):
        return self._set_contrast(*self.max_range)

    def _auto_contrast(self, event):
        return self._set_contrast(*self.auto_range)

    def _validate(self, event):
        self._save_data()
        plt.close(self.fig)

    def _save_data(self):
        """"""
        # The float is here because np.int64 and np.float64 do not work with JSON
        vmin, vmax = [float(self.widgets[s].val) for s in ('slider min', 'slider max')]
        self.parameter.vmin = vmin
        self.parameter.vmax = vmax

    def _process_image(self):
        """How to process image initially, define in subclasses."""
        pass

    def _create_imshow(self):
        """How to display the image and create an imshow object. Subclass."""
        pass

    def _get_init_range(self):
        """Return initial range vmin, vmax (auto if not existing yet)"""
        current_vmin = self.parameter.vmin
        current_vmax = self.parameter.vmax
        if current_vmin is None and current_vmax is None:
            return self.auto_range
        else:
            return current_vmin, current_vmax

    def _create_axes(self):
        self.axs = {}

        self.fig = plt.figure(figsize=(12, 5))

        self.axs['image'] = self.fig.add_axes([0.05, 0.05, 0.5, 0.9])
        self.imshow = self._create_imshow()

        self.axs['histogram'] = self.fig.add_axes([0.7, 0.45, 0.25, 0.5])
        self.axs['histogram'].hist(self.img_hist, bins='auto')
        self.axs['histogram'].set_xlim(self.max_range)

        vmin, vmax = self.init_range
        self.min_line = self.axs['histogram'].axvline(vmin, color='k')
        self.max_line = self.axs['histogram'].axvline(vmax, color='k')

        self.axs['slider min'] = self.fig.add_axes([0.7, 0.2, 0.25, 0.04])
        self.axs['slider max'] = self.fig.add_axes([0.7, 0.28, 0.25, 0.04])

        buttons = 'reset', 'full', 'auto', 'ok'
        xmin = 0.7        # start position from left
        xmax = 0.95       # final position
        ymin = 0.05       # start position from bottom
        h = 0.06          # button height
        dw = 0.01         # padding in width
        n = len(buttons)  # number of buttons
        w = (xmax - xmin - (n - 1) * dw) / n
        xs = [xmin + i * (w + dw) for i in range(n)]
        for button_name, x in zip(buttons, xs):
            name = 'button ' + button_name
            self.axs[name] = self.fig.add_axes([x, ymin, w, h])

    def _create_widgets(self):
        self.widgets = {}

        vmin, vmax = self.init_range
        vmin_min, vmax_max = self.max_range
        v_step = 1 if type(vmax_max) is int else None

        self.widgets['slider min'] = Slider(
            ax=self.axs['slider min'],
            label='min',
            valmin=vmin_min,
            valmax=vmax_max,
            valinit=vmin,
            valstep=v_step,
            color='steelblue',
            alpha=0.5,
        )

        self.widgets['slider max'] = Slider(
            ax=self.axs['slider max'],
            label='max',
            valmin=vmin_min,
            valmax=vmax_max,
            valinit=vmax,
            valstep=v_step,
            color='steelblue',
            alpha=0.5,
        )

        self.widgets['button reset'] = Button(self.axs['button reset'], 'Reset')
        self.widgets['button full'] = Button(self.axs['button full'], 'Full')
        self.widgets['button auto'] = Button(self.axs['button auto'], 'Auto')
        self.widgets['button ok'] = Button(self.axs['button ok'], 'OK')

    def run(self):
        """Start viewer / setter"""
        self._process_image()
        self._create_axes()
        self._create_widgets()

        self.widgets['slider min'].on_changed(self._update_min)
        self.widgets['slider max'].on_changed(self._update_max)

        self.widgets['button reset'].on_clicked(self._reset_contrast)
        self.widgets['button full'].on_clicked(self._full_contrast)
        self.widgets['button auto'].on_clicked(self._auto_contrast)
        self.widgets['button ok'].on_clicked(self._validate)

        # Without this return, the widgets tend to be garbage collected
        return self.widgets


class ContrastSetterViewer(DoubleSliderBase):
    """Viewer with double slider to set contrast (min/max display gray level)."""

    def __init__(self, img_series, num=0, **kwargs):
        """Interactively define brightness / contrast

        Parameters
        ----------
        img_series : ImgSeries object or equivalent (e.g. ImgStack).

        num : int
            image ('num' id) on which to define contrast. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        super().__init__(img_series=img_series, num=num, **kwargs)
        if not self.img_series.threshold.is_empty:
            raise RuntimeError('Impossible to set contrast on thresholded images')
        self.parameter = self.img_series.display

    def _update_min(self, value):
        self.imshow.norm.vmin = value
        self.min_line.set_xdata((value, value))

    def _update_max(self, value):
        self.imshow.norm.vmax = value
        self.max_line.set_xdata((value, value))

    def _process_image(self):
        self.img = self.img_series.read(num=self.num)
        self.img_hist = self.img[np.isfinite(self.img)].flatten()
        self.max_range = max_pixel_range(self.img)
        self.auto_range = self.img_hist.min(), self.img_hist.max()
        self.init_range = self._get_init_range()

    def _create_imshow(self):
        return self.img_series._imshow(
            self.img,
            ax=self.axs['image'],
            **self.kwargs,
        )


class ThresholdSetterViewer(DoubleSliderBase):
    """Viewer with double slider to set threshold (min / max)."""

    def __init__(self, img_series, num=0, **kwargs):
        """Interactively define brightness / contrast

        Parameters
        ----------
        img_series : ImgSeries object or equivalent (e.g. ImgStack).

        num : int
            image ('num' id) on which to define contrast. Note that
            this number can be different from the name written in the image
            filename, because num always starts at 0 in the first folder.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        super().__init__(img_series=img_series, num=num, **kwargs)
        self.parameter = self.img_series.threshold

    def _get_current_range(self):
        try:
            current_range = self.current_range
        except AttributeError:
            current_range = self.init_range
        return current_range

    def _update_min(self, value):
        _, vmax = self._get_current_range()
        self.current_range = value, vmax
        img = double_threshold(
            img=self.img_raw,
            vmin=value,
            vmax=vmax,
        )
        self.imshow.set_array(img)
        self.min_line.set_xdata((value, value))

    def _update_max(self, value):
        vmin, _ = self._get_current_range()
        self.current_range = vmin, value
        img = double_threshold(
            img=self.img_raw,
            vmin=vmin,
            vmax=value,
        )
        self.imshow.set_array(img)
        self.max_line.set_xdata((value, value))

    def _process_image(self):

        self.img_raw = self.img_series.read(num=self.num, threshold=False)
        self.img_hist = self.img_raw[np.isfinite(self.img_raw)].flatten()
        self.max_range = max_pixel_range(self.img_raw)

        vmin_auto = np.median(self.img_hist)
        _, vmax_auto = self.max_range
        self.auto_range = vmin_auto, vmax_auto
        self.init_range = self._get_init_range()

        self.img = double_threshold(
            self.img_raw,
            *self.init_range,
        )

    def _create_imshow(self):
        # Without the vmin/vmax arguments, the thresholded image is not
        # displayed properly because at this stage the img_series is not
        # thresholded yet and still has default vmin, vmax from before
        # thresholding.
        return self.img_series._imshow(
            self.img,
            ax=self.axs['image'],
            vmin=0, vmax=1,
        )
