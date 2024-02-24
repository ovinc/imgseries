"""Calculate and view intensity profile on grey or RGB images."""

import matplotlib.pyplot as plt
import numpy as np
from drapo import Line
from skimage import measure

from ..viewers import AnalysisViewer


class InteractiveLine(Line):

    def __init__(self, viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer

    def on_mouse_release(self, event):
        """Trigger re-plotting when Line is released"""
        super().on_mouse_release(event)
        if event.inaxes is not self.viewer.ax_img:
            return
        self.viewer._plot(num=self.viewer.num)

    def _on_right_pick(self):
        """To avoid deleting line with right click"""
        pass


class ProfileViewer(AnalysisViewer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Current img number, will be updated with self._get_data()
        self.num = None

    def _create_figure(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 15))
        self.ax_img, self.ax_profile = self.axs
        self.ax_profile.grid()

    def _get_data(self, num):
        """How to get image / analysis and other data to _plot for each frame.

        Input: image number (int)
        Output: data (arbitrary data format usable by _first_plot() and _update_plot())
        """
        data = {'num': num}

        # If image hasn't changed (just line position, no need to reload image);
        # The lines below are also exectuted when profile is first loaded
        # in order to define self.num and self.img for the first time
        if num != self.num:
            self.num = num
            # to update image on plot if img has changed
            self.img = self.analysis.img_series.read(num=num)
            data['new image'] = self.img

        return data

    def _first_plot(self, data):
        """What to do the first time data arrives on the _plot.

        self.updated_artists must be defined here.

        Input: data
        """
        self.imshow = self.analysis.img_series._imshow(
            self.img,
            ax=self.ax_img,
            **self.kwargs,
        )

        # This needs to be called AFTER the first imshow(), because
        # if not, it's the line which will set the limits of the image
        # (typically, 0 to 1)
        self.interactive_line = InteractiveLine(self, ax=self.ax_img)

        # Calculate and display first profile --------------------------------

        rr, levels = self.analysis._generate_profiles(
            img=self.img,
            line_position=self.interactive_line.get_position(),
            radius=self.analysis.radius,
        )

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
            img = data['new image']
        except KeyError:
            pass
        else:
            self.imshow.set_array(img)

        rr, levels = self.analysis._generate_profiles(
            img=self.img,  # self.img and not img, in case image has not been updated
            line_position=self.interactive_line.get_position(),
            radius=self.analysis.radius,
        )

        if len(self.profile_curves) > 1:
            for i, profile_curve in enumerate(self.profile_curves):
                profile_curve.set_data(rr, levels[:, i])
        else:
            profile_curve, = self.profile_curves
            profile_curve.set_data(rr, levels)

        self.ax_profile.relim()  # without this, axes limits change don't work
        self.ax_profile.autoscale(axis='both')

        # DO NOT REMOVE BELOW (if not present, line profile does not update
        # when line is moved and released by the user)
        self.fig.canvas.draw()


class Profile:

    def __init__(
        self,
        img_series,
        radius=1,
        viewer=ProfileViewer,
        **kwargs,
    ):
        self.Viewer = viewer
        self.img_series = img_series
        self.radius = radius
        self.kwargs = kwargs

    @staticmethod
    def _generate_profiles(img, line_position, radius):

        (x1, y1), (x2, y2) = line_position

        # For some reason, coordinates need to be reversed here between x/y
        # when calling the scikit-image function
        pt1 = (y1, x1)
        pt2 = (y2, x2)

        levels = measure.profile_line(img, src=pt1, dst=pt2, linewidth=radius)
        rr = np.arange(len(levels))

        return rr, levels

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
