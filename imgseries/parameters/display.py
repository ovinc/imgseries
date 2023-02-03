"""Classes to store display parameters: contrast, colormaps, etc.

These parameters are just for displaying in imshow(), but do not impact
the images themselves.
"""


# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Local imports
from .parameters_base import DisplayParameter


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
