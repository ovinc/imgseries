"""Calculate and view intensity profile on grey or RGB images."""

import matplotlib.pyplot as plt
import numpy as np
from drapo import Line

from .viewers import ImageViewer


class LineProfile(Line):

    def __init__(self, viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer

    def on_mouse_release(self, event):
        super().on_mouse_release(event)
        self.viewer._plot(num=self.viewer.num)


def _generate_level_lines(img, line_position, npts, radius):

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


class ProfileViewer(ImageViewer):

    def __init__(self, img_series, npts=100, radius=2, **kwargs):
        super().__init__()
        self.img_series = img_series
        self.npts = npts
        self.radius = radius
        self.kwargs = kwargs

    def _create_figure(self, num=0):
        self.fig, (self.ax_img, self.ax_profile) = plt.subplots(2, 1, figsize=(10, 15))
        self.axs = self.ax_img, self.ax_profile
        self.ax_profile.grid()

        self.num = num
        self.img = self.img_series.read(num=num)

        self.imshow = self.img_series._imshow(self.img,
                                              ax=self.ax_img,
                                              **self.kwargs)
        self.profile_line = LineProfile(self, ax=self.ax_img)

    def _get_data(self, num):
        """How to get image / analysis and other data to _plot for each frame.

        Input: image number (int)
        Output: data (arbitrary data format usable by _first_plot() and _update_plot())
        """
        # if image hasn't changed (just line position, no need to reload image)
        if num  != self.num:
            self.num = num
            img = self.img_series.read(num=num)
            self.img = img
            data = {'image': img}  # to update image on plot if img has changed

        line_position = self.profile_line.get_position()
        rr, levels = _generate_level_lines(img=self.img,
                                           line_position=line_position,
                                           npts=self.npts,
                                           radius=self.radius)

        return {'profiles': (rr, levels)}

    def _first_plot(self, data):
        """What to do the first time data arrives on the _plot.

        self.updated_artists must be defined here.

        Input: data
        """
        rr, levels = data['profiles']
        self.profiles = self.ax_profile.plot(rr, levels)
        if len(self.profiles) > 1:
            colors = ('r', 'g', 'b')
            for profile, color in zip(self.profiles, colors):
                profile.set_color(color)
        else:
            profile, = self.profiles
            profile.set_color('k')

        self.updated_artists = (self.imshow,) + self.profile_line.all_artists

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

        if len(self.profiles) > 1:
            for i, profile in enumerate(self.profiles):
                profile.set_data(rr, levels[:, i])
        else:
            profile, = self.profiles
            profile.set_data(rr, levels)

        self.ax_profile.relim()  # needed for autoscale to wotk
        self.ax_profile.autoscale(axis='both')
        self.fig.canvas.draw()
