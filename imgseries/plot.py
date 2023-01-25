"""Create live plots for analysis on image sequences"""

# Nonstandard
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


class ImagePlot:
    """Base class for plotting of images and additional data for animations."""

    def __init__(self, img_series, transform=True, **kwargs):
        """Parameters:

        - img_series: image series or analysis series (e.g. GreyLevel)
        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.
        - **kwargs: any useful kwargs to pass to imshow().
        """
        self.img_series = img_series
        self.transform = transform
        self.kwargs = kwargs
        self.plot_init_done = False

    def plot(self, num):
        """How to plot data during live views of analysis."""
        data = self.get_data(num)

        if not self.plot_init_done:
            self.first_plot(data)
            self.plot_init_done = True
        else:
            self.update_plot(data)

        return self.updated_artists

    def create_plot(self):
        """Define in subclass, has to define at least self.fig"""
        pass

    def get_data(self, num):
        """How to get image / analysis and other data to plot for each frame."""
        pass

    def first_plot(self, data):
        """What to do the first time data arrives on the plot.

        self.updated_artists must be defined here.
        """
        pass

    def update_plot(self, data):
        """What to do upon iterations of the plot after the first time."""
        pass

    def animate(self, nums, blit=False):
        """Animate an image plot with a FuncAnimation

        Parameters:
        - nums: frames to consider for the animation (iterable)
        - blit: if True, use blitting for fast rendering
        """
        self.create_plot()
        self.plot_init_done = False

        animation = FuncAnimation(fig=self.fig,
                                  func=self.plot,
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

        self.create_plot()
        self.plot_init_done = False

        self.plot(num=num_min)

        self.fig.subplots_adjust(bottom=0.1)
        ax_slider = self.fig.add_axes([0.1, 0.01, 0.8, 0.04])

        slider = Slider(ax=ax_slider,
                        label='#',
                        valmin=num_min,
                        valmax=num_max,
                        valinit=num_min,
                        valstep=num_step,
                        color='steelblue',
                        alpha=0.5)

        slider.on_changed(self.plot)

        return slider
