"""Transformations on images : crop, rotate, etc."""

# Non-standard modules
import matplotlib.pyplot as plt
import imgbasics
from imgbasics.cropping import _cropzone_draw


class TransformParameter:
    """Base class to define common methods for different transform classes."""

    parameter_type = None  # define in subclasses (e.g. "zones")

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. GreyLevel)
        """
        self.img_series = img_series  # ImgSeries object on which to define zones
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def load(self, filename=None):
        """Load transform data from .json file and put it in self.data.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        all_data = self.img_series.load_metadata(filename=filename)
        self.data = all_data[self.parameter_type]

    @property
    def is_empty(self):
        status = True if self.data == {} else False
        return status


class Zones(TransformParameter):
    """Class to store and manage areas of interest on series of images."""

    parameter_type = 'zones'

    def define(self, n=1, num=0, draggable=False):
        """Interactively define n zones image.

        Parameters
        ----------
        - n: number of zones to analyze (default 1)

        - num: image ('num' id) on which to select crop zones. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - draggable: use draggable rectangle from drapo to define crop zones
          instead of clicking to define opposite rectangle corners.

        Output
        ------
        None, but stores in self.data a dict with every cropzone used during
        the analysis, with:
        Keys: 'zone 1', 'zone 2', etc.
        Values: tuples (x, y, width, height)
        """
        img = self.img_series.read(num=num)

        if img.ndim > 2:
            kwargs = {}
        else:
            kwargs = {'cmap': 'gray'}

        fig, ax = plt.subplots()
        ax.imshow(img, **kwargs)
        ax.set_title('All zones defined so far')

        zones = {}

        for k in range(1, n + 1):

            msg = f'Select zone {k} / {n}'

            _, cropzone = imgbasics.imcrop(img,
                                           message=msg,
                                           draggable=draggable,
                                           **kwargs)

            name = f'zone {k}'
            zones[name] = cropzone
            _cropzone_draw(ax, cropzone, c='b')

        plt.close(fig)

        self.data = zones

    def show(self, num=0, **kwargs):
        """show the defined zones on image (image id num if specified)

        Parameters
        ----------
        - num: id number of image on which to show the zones (default first one).
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default)
        """
        ax = self.img_series.show(num, **kwargs)
        ax.set_title(f'Analysis Zones (img #{num})')

        for zone in self.data.values():
            _cropzone_draw(ax, zone, c='r')

        return ax
