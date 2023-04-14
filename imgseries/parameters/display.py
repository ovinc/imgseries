"""Classes to store display parameters: contrast, colormaps, etc.

These parameters are just for displaying in imshow(), but do not impact
the images themselves.
"""


# Non-standard modules
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Local imports
from .parameters_base import DisplayParameter
from ..viewers import ContrastSetterViewer


class Display(DisplayParameter):
    """Gathers all parameters for displaying images.

    For the moment, interactive options are limited to setting
    - contrast (vmin, vmax)
    - colormap (cmap)
    """

    parameter_type = 'display'

    def _define_contrast(self, num=0, **kwargs):
        """Interactively define brightness / contrast

        Parameters
        ----------
        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        setter = ContrastSetterViewer(self.img_series, num=num, **kwargs)
        return setter.run()

    def _define_colormap(self, num=0, **kwargs):
        """Interactively define colormap

        Parameters
        ----------
        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
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

    def define(self, param, num=0, **kwargs):
        """Interactively define display parameters.

        Parameters
        ----------
        - param: 'contrast' (vmin, vmax) or 'colormap' (cmap)

        - num: image ('num' id) on which to define contrast. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        if param == 'contrast':
            return self._define_contrast(num=num, **kwargs)
        elif param == 'colormap':
            return self._define_colormap(num=num, **kwargs)
        else:
            raise ValueError(f'Unknown parameter: {param}')

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
    def vlims(self):
        return self.vmin, self.vmax

    @vlims.setter
    def vlims(self, value):
        self.vmin, self.vmax = value

    @property
    def cmap(self):
        try:
            return self.data['cmap']
        except KeyError:
            return

    @cmap.setter
    def cmap(self, value):
        self.data['cmap'] = value
