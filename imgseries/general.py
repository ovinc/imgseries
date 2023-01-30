"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path

# Nonstandard
import matplotlib.pyplot as plt
from skimage import io
import filo

# local imports
from .config import filenames, _to_json, _from_json
from .config import _read, _rgb_to_grey, _rotate, _crop
from .image_parameters import Rotation, Crop, Contrast, Colors
from .viewers import ImgSeriesViewer, ViewerTools


class ImgSeries(filo.Series, ViewerTools):
    """Class to manage series of images, possibly in several folders."""

    # Only for __repr__ (str representation of class object, see filo.Series)
    name = 'Image Series'

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = filenames['files'] + '.tsv'

    def __init__(self,
                 paths='.',
                 extension='.png',
                 savepath='.',
                 stack=None):
        """Init image series object.

        Parameters
        ----------
        - paths can be a string, path object, or a list of str/paths if data
          is stored in multiple folders.

        - extension: extension of files to consider (e.g. '.png')

        - savepath: folder in which to save parameters (transform, display etc.)

        If file series is in a stack rather than in a series of images:
        - stack: path to the stack (.tiff) file
          (parameters paths & extension will be ignored)
        """
        # Image transforms that are applied to all images of the series.
        self.rotation = Rotation(self)
        self.crop = Crop(self)

        # Display options (do not impact analysis)
        self.contrast = Contrast(self)
        self.colors = Colors(self)

        # Done here because self.stack will be an array, and bool(array)
        # generates warnings / errors
        self.is_stack = bool(stack)

        if self.is_stack:
            self.stack_path = Path(stack)
            self.stack = io.imread(stack, plugin="tifffile")
            self.savepath = Path(savepath)
        else:
            # Inherit useful methods and attributes for file series
            # (including self.savepath)
            filo.Series.__init__(self,
                                 paths=paths,
                                 extension=extension,
                                 savepath=savepath)

        ViewerTools.__init__(self, Viewer=ImgSeriesViewer)

        img = self.read()
        self.ndim = img.ndim

    def _rotate(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return _rotate(img, angle=self.rotation.data['angle'])

    def _crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return _crop(img, self.crop.data['zone'])

    def _set_substack(self, start, end, skip):
        """Generate subset of image numbers to be displayed/analyzed."""
        if self.is_stack:
            npts, *_ = self.stack.shape
            all_nums = list(range(npts))
            nums = all_nums[start:end:skip]
        else:
            files = self.files[start:end:skip]
            nums = [file.num for file in files]
        return nums

    def _get_imshow_kwargs(self):
        """Define kwargs to pass to imshow (to have grey by default for 2D)."""

        if not self.contrast.is_empty:
            kwargs = {**self.contrast.data}
        else:
            kwargs = {}

        if not self.colors.is_empty:
            kwargs = {**kwargs, **self.colors.data}
        elif self.ndim < 3:
            kwargs = {**kwargs, 'cmap': 'gray'}

        return kwargs

    def _imshow(self, img, ax=None, **kwargs):
        """Use plt.imshow() with default kwargs and/or additional ones

        Parameters
        ----------
        - img: image to display (numpy array or equivalent)

        - ax: axes in which to display the image. If not specified, create new
              ones

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        if ax is None:
            _, ax = plt.subplots()
        default_kwargs = self._get_imshow_kwargs()
        kwargs = {**default_kwargs, **kwargs}
        return ax.imshow(img, **kwargs)

    @staticmethod
    def rgb_to_grey(img):
        """"Convert RGB to grayscale"""
        return _rgb_to_grey(img)

    def read(self, num=0, transform=True):
        """Load image data (image identifier num across folders).

        By default, if transforms are defined on the image (rotation, crop)
        then they are applied here. Put transform=False to only load the raw
        image in the stack.
        """
        if not self.is_stack:
            img = _read(self.files[num].file)
        else:
            img = self.stack[num]

        if not transform:
            return img

        if not self.rotation.is_empty:
            img = self._rotate(img)

        if not self.crop.is_empty:
            img = self._crop(img)

        return img

    def load_transform(self, filename=None):
        """Load transform parameters (crop, rotation, etc.) from json file.

        Transforms are applied and stored in self.rotation, self.crop, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.rotation.reset()
        self.crop.reset()

        fname = filenames['transform'] if filename is None else filename
        transform_data = _from_json(self.savepath, fname)

        self.rotation.data = transform_data['rotation']
        self.crop.data = transform_data['crop']

    def save_transform(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = filenames['transform'] if filename is None else filename
        transform_data = {'rotation': self.rotation.data,
                          'crop': self.crop.data}
        _to_json(transform_data, self.savepath, fname)

    def load_display(self, filename=None):
        """Load display parameters (contrast, colormapn etc.) from json file.

        Display options are applied and stored in self.contrast, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.contrast.reset()
        self.colors.reset()

        fname = filenames['display'] if filename is None else filename
        display_data = _from_json(self.savepath, fname)

        self.contrast.data = display_data['contrast']
        self.colors.data = display_data['colors']

    def save_display(self, filename=None):
        """Save  display parameters (contrast, colormapn etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = filenames['display'] if filename is None else filename
        display_data = {'contrast': self.contrast.data,
                        'colors': self.colors.data}
        _to_json(display_data, self.savepath, fname)
