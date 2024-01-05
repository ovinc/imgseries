"""Calculate and view intensity profile on grey or RGB images."""

import matplotlib.pyplot as plt
import numpy as np
from drapo import Line

from .viewers import AnalysisViewer


class InteractiveLine(Line):

    def __init__(self, viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer

    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        if event.inaxes is not self.viewer.ax_img:
            return
        self.viewer._plot(num=self.viewer.num)


class ProfileViewer(AnalysisViewer):

    def _create_figure(self, num=0):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 15))
        self.ax_img, self.ax_profile = self.axs
        self.ax_profile.grid()

        self.num = num
        self.img = self.analysis.img_series.read(num=num)

        self.imshow = self.analysis.img_series._imshow(self.img,
                                                       ax=self.ax_img,
                                                       **self.kwargs)

        self.interactive_line = InteractiveLine(self, ax=self.ax_img)

    def _get_data(self, num):
        """How to get image / analysis and other data to _plot for each frame.

        Input: image number (int)
        Output: data (arbitrary data format usable by _first_plot() and _update_plot())
        """
        data = {}

        # if image hasn't changed (just line position, no need to reload image)
        if num != self.num:
            self.num = num
            img = self.analysis.img_series.read(num=num)
            self.img = img
            data['image'] = img  # to update image on plot if img has changed

        profiles = self.analysis._generate_profiles(
            img=self.img,
            line_position=self.interactive_line.get_position(),
            npts=self.analysis.npts,
            radius=self.analysis.radius,
        )

        data['profiles'] = profiles

        return data

    def _first_plot(self, data):
        """What to do the first time data arrives on the _plot.

        self.updated_artists must be defined here.

        Input: data
        """
        rr, levels = data['profiles']
        self.profile_curves = self.ax_profile.plot(rr, levels)
        if len(self.profile_curves) > 1:
            colors = ('r', 'g', 'b')
            for profile_curve, color in zip(self.profile_curves, colors):
                profile_curve.set_color(color)
        else:
            profile_curve, = self.profile_curves
            profile_curve.set_color('k')

        self.updated_artists = (self.imshow,) + self.interactive_line.all_artists

    def _update_plot(self, data):
        """What to do upon iterations of the _plot after the first time.

        Input: data
        """
        try:
            img = data['image']
        except KeyError:
            pass
        else:
            self.imshow.set_array(img)

        rr, levels = data['profiles']

        if len(self.profile_curves) > 1:
            for i, profile_curve in enumerate(self.profile_curves):
                profile_curve.set_data(rr, levels[:, i])
        else:
            profile_curve, = self.profile_curves
            profile_curve.set_data(rr, levels)

        self.ax_profile.relim()  # without this, axes limits change don't work
        self.ax_profile.autoscale(axis='both')
        self.fig.canvas.draw()


class Profile:

    def __init__(
        self,
        img_series,
        npts=100,
        radius=2,
        viewer=ProfileViewer,
        **kwargs,
    ):
        self.Viewer = viewer
        self.img_series = img_series
        self.npts = npts
        self.radius = radius
        self.kwargs = kwargs

    @staticmethod
    def _generate_profiles(img, line_position, npts, radius):

        img_shape = img.shape[:-1] if img.ndim > 2 else img.shape
        ii, jj = np.indices(img_shape)

        (x1, y1), (x2, y2) = line_position

        xx = np.linspace(x1, x2, num=npts)
        yy = np.linspace(y1, y2, num=npts)

        rr = np.hypot(xx - xx[0], yy - yy[0])

        def calculate_local_level(x, y):
            dx = ii - y  # for some reason, x and y have to be reversed here
            dy = jj - x
            dd = np.hypot(dx, dy)
            return img[dd < radius].mean(axis=0)

        levels = []
        for x, y in zip(xx, yy):
            level = calculate_local_level(x, y)
            levels.append(level)

        return rr, np.array(levels)

    # ==================== Interactive inspection methods ====================

    # Note: Initially, these were in a ViewerTools subclass to avoid code
    # repetition, but I eventually preferred to repeat code to avoid
    # multiple inheritance and weird couplings.

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series

        - transform: if True (default), apply active transforms
                     if False, load raw image.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.show(num=num)

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image series.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self.img_series._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=nums)

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - blit: if True, use blitting for faster animation.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self.img_series._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.animate(nums=nums, blit=blit)
