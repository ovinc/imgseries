"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path
from functools import lru_cache


# Nonstandard
import matplotlib.pyplot as plt
import filo
from skimage import io


# local imports
from .config import CONFIG
from .general import FileManager, ImageManager
from .image_parameters import Grayscale, Rotation, Crop, Filter, Subtraction
from .image_parameters import Contrast, Colors
from .viewers import ImgSeriesViewer, ViewerTools


class ImgSeries(filo.Series, ViewerTools):
    """Class to manage series of images, possibly in several folders."""

    # Only for __repr__ (str representation of class object, see filo.Series)
    name = 'Image Series'

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = CONFIG['filenames']['files'] + '.tsv'

    cache = False   # cache images during read() or not (changed in ImgSeriesCached)

    def __init__(self,
                 paths='.',
                 extension='.png',
                 savepath='.',
                 stack=None,
                 image_manager=ImageManager,
                 file_manager=FileManager):
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

        - image_manager: class (or object) that defines how to read and
                         transform images

        - file_manager: class (or object) that defines how to interact with
                        saved files
        """
        self.image_manager = image_manager
        self.file_manager = file_manager

        # Image transforms that are applied to all images of the series.
        self.grayscale = Grayscale(self)
        self.rotation = Rotation(self)
        self.crop = Crop(self)
        self.filter = Filter(self)
        self.subtraction = Subtraction(self)

        # Link image transform names to actual functions that apply them
        self.transforms = {'grayscale': self._rgb_to_grey,
                           'rotation': self._rotate,
                           'crop': self._crop,
                           'filter': self._filter,
                           'subtraction': self._subtract,
                           }

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
        self.initial_ndim = img.ndim
        self.ndim = self.initial_ndim

    def __repr__(self):
        if not self.is_stack:
            return super().__repr__()
        else:
            return f"{self.__class__.name}, file '{self.stack_path}', savepath '{self.savepath}'"

    # ========================== Global transforms ===========================

    def _rotate(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return self.image_manager.rotate(img,
                                         angle=self.rotation.data['angle'])

    def _crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return self.image_manager.crop(img,
                                       self.crop.data['zone'])

    def _filter(self, img):
        """Crop image according to pre-defined crop parameters"""
        return self.image_manager.filter(img,
                                         filter_type=self.filter.data['type'],
                                         size=self.filter.data['size'])

    def _subtract(self, img):
        """Subtract pre-set reference image to current image."""
        img_ref = self.subtraction.reference_image
        print(self.subtraction.relative)
        return self.image_manager.subtract(img,
                                           img_ref,
                                           relative=self.subtraction.relative)

    def _rgb_to_grey(self, img):
        """"Convert RGB to grayscale"""
        return self.image_manager.rgb_to_grey(img)


    def _apply_transform(self, img, **kwargs):
        """Apply stored transforms on the image (crop, rotation, etc.)"""

        for transform_name in CONFIG['image transforms']:

            transform_object = getattr(self, transform_name)      # e.g. self.rotation
            transform_function = self.transforms[transform_name]  # e.g. self._rotate

            if not transform_object.is_empty and kwargs.get(transform_name, True):
                img = transform_function(img)

        return img

    # ============================= Misc. tools ==============================

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

    # ============================ Public methods ============================

    def read(self, num=0, transform=True, **kwargs):
        """Load image data as an array.

        Parameters
        ----------

        - num: image identifier (integer)

        - transform: By default, if transforms are defined on the image
                     (rotation, crop etc.), then they are applied here.
                     Put transform=False to only load the raw image in the
                     stack.

        - kwargs: by default if transform=True, all active transforms are
                  applied. Set any transform name to False to not apply
                  this particular transform.
                  e.g. images.read(subtraction=False)
        """
        if not self.is_stack:
            img = self.image_manager.read(self.files[num].file)
        else:
            img = self.stack[num]

        if transform:
            return self._apply_transform(img, **kwargs)
        else:
            return img

    def load_transform(self, filename=None):
        """Load transform parameters (crop, rotation, etc.) from json file.

        Transforms are applied and stored in self.rotation, self.crop, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        transform_data = self.file_manager.from_json(self.savepath, fname)

        for transform_name in CONFIG['image transforms']:
            transform_object = getattr(self, transform_name)
            transform_object.data = transform_data.get(transform_name, {})
            transform_object._update_parameters()

    def save_transform(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename

        transform_data = {transform_name: getattr(self, transform_name).data
                          for transform_name in CONFIG['image transforms']}

        self.file_manager.to_json(transform_data, self.savepath, fname)

    def load_display(self, filename=None):
        """Load display parameters (contrast, colormapn etc.) from json file.

        Display options are applied and stored in self.contrast, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.contrast.reset()
        self.colors.reset()

        fname = CONFIG['filenames']['display'] if filename is None else filename
        display_data = self.file_manager.from_json(self.savepath, fname)

        self.contrast.data = display_data['contrast']
        self.colors.data = display_data['colors']

    def save_display(self, filename=None):
        """Save  display parameters (contrast, colormapn etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        display_data = {'contrast': self.contrast.data,
                        'colors': self.colors.data}
        self.file_manager.to_json(display_data, self.savepath, fname)


def series(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option."""
    if not cache:

        return ImgSeries(*args, **kwargs)

    else:

        class ImgSeriesCached(ImgSeries):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgSeriesCached(*args, **kwargs)
