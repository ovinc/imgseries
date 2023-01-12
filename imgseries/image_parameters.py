"""Classes to store various parameters: crop zones, rotations, contours, etc."""

# Non-standard modules
import matplotlib.pyplot as plt
import imgbasics
from imgbasics.cropping import _cropzone_draw


class ImageParameter:
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

    def _get_imshow_kwargs(self, img):
        """Define kwargs to pass to imshow (to have grey by default for 2D)"""
        return {'cmap': 'gray'} if img.ndim < 3 else {}

    def load(self, filename=None):
        """Load parameter data from .json file and put it in self.data.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        all_data = self.img_series.load_metadata(filename=filename)
        self.data = all_data[self.parameter_type]

    @property
    def is_empty(self):
        return not self.data


class Zones(ImageParameter):
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
        kwargs = self._get_imshow_kwargs(img)

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
        (note: cmap is grey by default for 2D images, see ImgSeries.show())
        """
        ax = self.img_series.show(num, **kwargs)
        ax.set_title(f'Analysis Zones (img #{num})')

        for zone in self.data.values():
            _cropzone_draw(ax, zone, c='r')

        return ax


class Contours(ImageParameter):
    """Class to store and manage reference contours param in image series."""

    parameter_type = 'contours'

    def define(self, level, n=1, num=0):
        """Interactively define n contours on an image at level level.

        Parameters
        ----------
        - level: grey level at which to define threshold to detect contours
        - n: number of contours
        - num: image identifier (num=0 corresponds to first image in first folder)

        Output
        ------
        None, but stores in self.data a dictionary with keys:
        'crop', 'position', 'level', 'image'
        """
        img = self.img_series.read(num=num)
        kwargs = self._get_imshow_kwargs(img)

        img_crop, crop = imgbasics.imcrop(img, **kwargs)
        contours = self.img_series._find_contours(img_crop, level)

        # Display the cropped image and plot all contours found --------------

        fig, ax = plt.subplots()
        ax.imshow(img_crop, **kwargs)
        ax.set_xlabel('Left click on vicinity of contour to select.')

        for contour in contours:
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        # Interactively select contours of interest on image -----------------

        positions = {}

        for k in range(1, n + 1):

            ax.set_title(f'Contour {k} / {n}')
            fig.canvas.draw()
            fig.canvas.flush_events()

            pt, = plt.ginput()

            contour = imgbasics.closest_contour(contours, pt, edge=True)
            x, y = imgbasics.contour_coords(contour, source='scikit')

            ax.plot(x, y, linewidth=1, c='y')
            plt.pause(0.01)

            xc, yc = imgbasics.contour_properties(x, y)['centroid']

            name = f'contour {k}'
            positions[name] = (xc, yc)  # store position of centroid

        plt.close(fig)

        self.data = {'crop': crop,
                     'position': positions,
                     'level': level,
                     'image': num}

    def show(self, **kwargs):
        """Show reference contours used for contour tracking.

        Parameters
        ----------
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default for images with 1 color channel)
        """
        num = self.data['image']
        level = self.data['level']
        crop = self.data['crop']
        positions = self.data['position']

        # Load image, crop it, and calculate contours
        img = self.img_series.read(num)
        img_crop = imgbasics.imcrop(img, crop)
        contours = self.img_series._find_contours(img_crop, level)

        fig, ax = plt.subplots()

        if 'cmap' not in kwargs and img.ndim < 3:
            kwargs['cmap'] = 'gray'
        ax.imshow(img_crop, **kwargs)

        # Find contours closest to reference positions and plot them
        for contour in contours:
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=1, c='b')

        # Interactively select contours of interest on image -----------------
        for pt in positions.values():
            contour = imgbasics.closest_contour(contours, pt, edge=False)
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        ax.set_title(f'img #{num}, grey level {level}')

        plt.show()

        return ax

