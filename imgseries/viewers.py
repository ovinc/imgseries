"""Create live plots for analysis on image sequences"""

# Nonstandard
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Local imports
from .managers import max_pixel_range


class KeyPressSlider(Slider):
    """Slider to inspect images, with keypress response"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connect_events()
        if self.valstep is None:
            self.keystep = (self.valmax - self.valmin) * 0.01
        else:
            self.keystep = self.valstep

    def _increase_val(self, nstep=1):
        """Increase value by a nstep steps; got to beginning if > valmax"""
        new_val = self.val + nstep * self.keystep
        if new_val > self.valmax:
            new_val = self.valmin
        self.set_val(new_val)

    def _decrease_val(self, nstep=1):
        """Increase value by a nstep steps; got to beginning if > valmax"""
        new_val = self.val - nstep * self.keystep
        if new_val < self.valmin:
            new_val = self.valmax
        self.set_val(new_val)

    def _connect_events(self):
        self.cid_keypressk = self.ax.figure.canvas.mpl_connect(
            'key_press_event',
            self._on_key_press,
        )

    def _on_key_press(self, event):
        if event.key == 'right':
            self._increase_val()
        if event.key == 'left':
            self._decrease_val()
        if event.key == 'up':
            self._increase_val(nstep=10)
        if event.key == 'down':
            self._decrease_val(nstep=10)


class ImageViewer:
    """Base class for plotting of images and additional data for animations."""

    def __init__(self):
        """Init Image Viewer"""
        self.plot_init_done = False

    def _create_figure(self):
        """Define in subclass, has to define at least self.fig., and self.axs
        if self.axs is not defined in self._first_plot()
        """
        pass

    def _connect_events(self):
        """Called after _create_figure() to connect events, e.g. figure closing"""
        pass

    def _get_data(self, num):
        """How to get image / analysis and other data to _plot for each frame.

        Input: image number (int)
        Output: data (arbitrary data format usable by _first_plot() and _update_plot())
        """
        pass

    def _first_plot(self, data):
        """What to do the first time data arrives on the _plot.

        self.updated_artists must be defined here.
        self.axs must be defined here, except if done in self._create_figure()

        Input: data
        """
        pass

    def _update_plot(self, data):
        """What to do upon iterations of the _plot after the first time.

        Input: data
        """
        pass

    def _plot(self, num):
        """How to _plot data during live views of analysis."""
        data = self._get_data(num)

        if not self.plot_init_done:
            self._first_plot(data)
            self.plot_init_done = True
        else:
            self._update_plot(data)

        return self.updated_artists

    def show(self, num=0):
        """Show a single, non-animated image (num: image number)."""
        self._create_figure()
        self._connect_events()
        self._plot(num=num)
        return self.axs

    def animate(self, nums, blit=False):
        """Animate an image _plot with a FuncAnimation

        Parameters:
        - nums: frames to consider for the animation (iterable)
        - blit: if True, use blitting for fast rendering
        """
        self._create_figure()
        self._connect_events()
        self.plot_init_done = False

        animation = FuncAnimation(
            fig=self.fig,
            func=self._plot,
            frames=nums,
            cache_frame_data=False,
            repeat=False,
            blit=blit,
            init_func=lambda: None,  # prevents calling twice the first num
        )

        return animation

    def inspect(self, nums):
        """Inspect image series with a slider.

        Parameters:
        - nums: frames to consider for the animation (iterable)
        """
        num_min = min(nums)
        num_max = max(nums)

        if num_max > num_min:  # avoids division by 0 error when just 1 image
            num_step = (num_max - num_min) // (len(nums) - 1)
        else:
            num_step = 1

        self._create_figure()
        self._connect_events()
        self.plot_init_done = False

        self._plot(num=num_min)

        self.fig.subplots_adjust(bottom=0.1)
        ax_slider = self.fig.add_axes([0.1, 0.01, 0.8, 0.03])

        slider = KeyPressSlider(
            ax=ax_slider,
            label='#',
            valmin=num_min,
            valmax=num_max,
            valinit=num_min,
            valstep=num_step,
            color='steelblue',
            alpha=0.5,
        )

        slider.on_changed(self._plot)

        return slider


class ImgSeriesViewer(ImageViewer):
    """Matplotlib viewer to inspect image series (no analysis).

    See ImageViewer for details.
    """

    def __init__(self, img_series, transform=True, **kwargs):
        """Parameters
           ----------

        - img_series: image series or analysis series (e.g. GreyLevel)

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

        - kwargs: any keyword-argument to pass to imshow().
        """
        self.img_series = img_series
        self.transform = transform
        self.kwargs = kwargs
        super().__init__()

    def _create_figure(self):
        self.fig, self.ax = plt.subplots()
        self.axs = self.ax,

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

    def _display_info(self, data):
        num = data['num']
        if self.img_series.is_stack:
            title = 'Image'
        else:
            title = self.img_series.files[num].name
        raw_info = ' [RAW]' if not self.transform else ''
        self.ax.set_title(f'{title} (#{num}){raw_info}')


class AnalysisViewer(ImageViewer):
    """Matplotlib viewer to display analysis results alongside images."""

    def __init__(self, analysis, live=False, **kwargs):
        """Parameters
           ----------

        - analysis: analysis object (e.g. GreyLevel(), ContourTracking(), etc.)

        - live: if True, get data in real time from analysis being made.

        - kwargs: any keyword-argument to pass to imshow().

        Note: transform=False not available here because analysis classes use
        transformed images.
        """
        self.analysis = analysis
        self.live = live
        self.kwargs = kwargs
        super().__init__()

    def _get_data(self, num):
        """Analyses classes should define adequate methods if needed:

        - Analysis.__analyze_live() to get real time data from analysis
        - Analysis._generate_data() to get usable data from stored data."""
        if self.live:
            return self.analysis._analyze_live(num)
        else:
            return self.analysis.formatter._generate_data(num)

    def _connect_events(self):
        """Connect figure events"""
        self.cid_close = self.fig.canvas.mpl_connect(
            'close_event',
            self._on_fig_close,
        )

    def _on_fig_close(self, event):
        """This is because we want the analysis (i.e. animation) to finish
        before saving the data in live mode."""
        if self.live:
            self.analysis.formatter._to_results()


# ==================== Interactive setting of parameters =====================


class DoubleSliderBase:
    """Viewer with double slider to set contrast/threshold"""

    def __init__(self, img_series, num=0, **kwargs):
        """Interactively define brightness / contrast or threshold

        Parameters
        ----------
        - img_series: ImgSeries object or equivalent.

        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
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
        - img_series: ImgSeries object or equivalent.

        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
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
        - img_series: ImgSeries object or equivalent.

        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
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
        img = self.img_series.img_transformer.img_manager.threshold(
            img=self.img_raw,
            vmin=value,
            vmax=vmax,
        )
        self.imshow.set_array(img)
        self.min_line.set_xdata((value, value))

    def _update_max(self, value):
        vmin, _ = self._get_current_range()
        self.current_range = vmin, value
        img = self.img_series.img_transformer.img_manager.threshold(
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

        self.img = self.img_series.img_transformer.img_manager.threshold(
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
