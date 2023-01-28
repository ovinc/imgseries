"""Create live plots for analysis on image sequences"""

# Nonstandard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class ImageViewer:
    """Base class for plotting of images and additional data for animations."""

    def __init__(self):
        """Init Image Viewer"""
        self.plot_init_done = False

    def _create_figure(self):
        """Define in subclass, has to define at least self.fig"""
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

    def show(self, num):
        """Show a single, non-animated image (num: image number)."""
        self._create_figure()
        self._plot(num=num)
        return self.ax

    def animate(self, nums, blit=False):
        """Animate an image _plot with a FuncAnimation

        Parameters:
        - nums: frames to consider for the animation (iterable)
        - blit: if True, use blitting for fast rendering
        """
        self._create_figure()
        self.plot_init_done = False

        animation = FuncAnimation(fig=self.fig,
                                  func=self._plot,
                                  frames=nums,
                                  cache_frame_data=False,
                                  repeat=False,
                                  blit=blit)

        return animation

    def inspect(self, nums):
        """Inspect image series with a slider.

        Parameters:
        - nums: frames to consider for the animation (iterable)
        """
        num_min = min(nums)
        num_max = max(nums)
        num_step = (num_max - num_min) // (len(nums) - 1)

        self._create_figure()
        self.plot_init_done = False

        self._plot(num=num_min)

        self.fig.subplots_adjust(bottom=0.1)
        ax_slider = self.fig.add_axes([0.1, 0.01, 0.8, 0.03])

        slider = Slider(ax=ax_slider,
                        label='#',
                        valmin=num_min,
                        valmax=num_max,
                        valinit=num_min,
                        valstep=num_step,
                        color='steelblue',
                        alpha=0.5)

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

    def _get_data(self, num):
        img = self.img_series.read(num, transform=self.transform)
        return {'num': num, 'image': img}

    def _first_plot(self, data):
        self.imshow = self.img_series._imshow(data['image'],
                                              ax=self.ax,
                                              **self.kwargs)
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

    def __init__(self, analysis, live=False, transform=True, **kwargs):
        """Parameters
           ----------

        - analysis: analysis object (e.g. GreyLevel(), ContourTracking(), etc.)

        - live: if True, get data in real time from analysis being made.

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

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
        - Analysis._regenerate_data() to get usable data from stored data."""
        if self.live:
            return self.analysis._analyze_live(num)
        else:
            return self.analysis._regenerate_data(num)


class ViewerTools:
    """Class that adds viewing tools to ImgSeries or Analysis classes."""

    def __init__(self, Viewer):
        """Define which viewer is used to display images and data."""
        self.Viewer = Viewer

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, load raw image.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.show(num=num)

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=nums)

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

        - blit: if True, use blitting for faster animation.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.animate(nums=nums, blit=blit)
