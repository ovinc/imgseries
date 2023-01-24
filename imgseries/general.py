"""Class ImgSeries for image series manipulation"""

# Standard library imports
import json
from pathlib import Path

# Nonstandard
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
import filo

# local imports
from .config import filenames
from .config import _read, _rgb_to_grey, _rotate, _crop
from .image_parameters import Rotation, Crop


# ================ General classes for managing image series =================


class ImgSeries(filo.Series):
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

        - savepath: folder in which to save parameter / analysis data.

        If file series is in a stack rather than in a series of images:
        - stack: path to the stack (.tiff) file
          (parameters paths & extension will be ignored)
        """
        # Image transforms that are applied to all images of the series.
        self.rotation = Rotation(self)
        self.crop = Crop(self)

        # Done here because self.stack will be an array, and bool(array)
        # generates warnings / errors
        self.is_stack = bool(stack)

        if self.is_stack:
            self.stack_path = Path(stack)
            self.stack = io.imread(stack, plugin="tifffile")
            self.savepath = Path(savepath)
            self.total_img_number, *_ = self.stack.shape
        else:
            # Inherit useful methods and attributes for file series
            # (including self.savepath)
            super().__init__(paths=paths,
                             extension=extension,
                             savepath=savepath)
            self.total_img_number = len(self.files)

    def _rotate(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return _rotate(img, angle=self.rotation.data['angle'])

    def _crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return _crop(img, self.crop.data['zone'])

    def _from_json(self, filename):
        """"Load json file"""
        file = self.savepath / (filename + '.json')
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def _to_json(self, data, filename):
        """"Save data (dict) to json file"""
        file = self.savepath / (filename + '.json')
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

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

        if transform and not self.rotation.is_empty:
            img = self._rotate(img)

        if transform and not self.crop.is_empty:
            img = self._crop(img)

        return img

    def load_transform(self, filename=None):
        """Return transform parameters (crop, rotation, etc.)
        from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.rotation.reset()
        self.crop.reset()

        fname = filenames['transform'] if filename is None else filename
        transform_data = self._from_json(fname)

        self.rotation.data = transform_data['rotation']
        self.crop.data = transform_data['crop']

    def save_transform(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file."""
        fname = filenames['transform'] if filename is None else filename
        transform_data = {'rotation': self.rotation.data,
                          'crop': self.crop.data}
        self._to_json(transform_data, fname)

    @staticmethod
    def rgb_to_grey(img):
        """"Convert RGB to grayscale"""
        return _rgb_to_grey(img)

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series
        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, load raw image.
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default for images with 1 color channel)
        """
        img = self.read(num, transform=transform)
        fig, ax = plt.subplots()

        if 'cmap' not in kwargs and img.ndim < 3:
            kwargs['cmap'] = 'gray'

        ax.imshow(img, **kwargs)

        title = 'Image' if self.is_stack else self.files[num].name
        raw_info = ' [RAW]' if not transform else ''

        ax.set_title(f'{title} (#{num}){raw_info}')
        ax.axis('off')

        return ax

    def inspect(self, **kwargs):
        """Interactively inspect image stack."""
        fig, ax = plt.subplots()

        img = self.read()
        if 'cmap' not in kwargs and img.ndim < 3:
            kwargs['cmap'] = 'gray'

        imshow = ax.imshow(img, **kwargs)
        fig.subplots_adjust(bottom=0.1)

        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.04])
        slider = Slider(ax=ax_slider,
                        label='#',
                        valmin=0,
                        valinit=0,
                        valmax=self.total_img_number - 1,
                        valstep=1,
                        color='steelblue', alpha=0.5)

        def update(num):
            imshow.set_array(self.read(num))

        slider.on_changed(update)

        return slider
