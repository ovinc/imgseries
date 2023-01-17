"""General tools for image anlysis."""


# Standard library imports
import json
import os
from pathlib import Path

# Nonstandard
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import filo
import gittools
import imgbasics
from imgbasics.transform import rotate

# local imports
from .config import filenames, csv_separator, checked_modules
from .config import _read, _rgb_to_grey
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

        self.is_stack = bool(stack)

        if self.is_stack:

            self.stack = io.imread(stack, plugin="tifffile")
            self.savepath = Path(savepath)

            # Below, pre-populate parameters to be saved as metadata. Other metadata
            # will be added to this dict before saving into metadata file

            stack_path = os.path.relpath(Path(stack), self.savepath)
            self.parameters = {'stack': stack_path}

        else:

            # Inherit useful methods for file series

            super().__init__(paths=paths, extension=extension, savepath=savepath)

            # Pre-populate metadata & parameters as explained above

            folders = [os.path.relpath(f, self.savepath) for f in self.folders]

            self.parameters = {'path': str(self.savepath.resolve()),
                               'folders': folders}

    def _rotate(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return rotate(img,
                      angle=self.rotation.data['angle'],
                      resize=True,
                      order=3)

    def _crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return imgbasics.imcrop(img, self.crop.data['zone'])

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
        Then they are applied here. Put transform=False to only load the raw
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


class Analysis:
    """Tools for analysis subclasses. Used as multiple inheritance."""

    def __init__(self, measurement_type=None):
        """Parameters:

        - measurement_type: specify 'glevel' or 'ctrack'
        """
        self.measurement_type = measurement_type  # for data loading/saving
        self.data = None      # Data that will be saved in analysis file

    # Tools for analysis subclasses ==========================================

    def load(self, filename=None):
        """Load analysis data from tsv file and return it as pandas DataFrame.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.tsv.
        """
        name = filenames[self.measurement_type] if filename is None else filename
        analysis_file = self.savepath / (name + '.tsv')
        data = pd.read_csv(analysis_file, index_col='num', sep=csv_separator)
        return data

    def load_metadata(self, filename=None):
        """Return analysis metadata from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        name = filenames[self.measurement_type] if filename is None else filename
        return self._from_json(name)

    def save(self, filename=None):
        """Save analysis data and metadata into .tsv / .json files.

        Parameters
        ----------
        filename:

            - If filename is not specified, use default filenames.

            - If filename is specified, it must be an str without the extension
              e.g. filename='Test' will create Test.tsv and Test.json files,
              containing tab-separated data file and metadata file, respectively.
        """
        name = filenames[self.measurement_type] if filename is None else filename
        analysis_file = self.savepath / (name + '.tsv')
        metadata_file = self.savepath / (name + '.json')

        # save analysis data -------------------------------------------------
        self.data.to_csv(analysis_file, sep=csv_separator)

        # save analysis metadata ---------------------------------------------
        gittools.save_metadata(file=metadata_file, info=self.parameters,
                               module=checked_modules,
                               dirty_warning=True, notag_warning=True,
                               nogit_ok=True, nogit_warning=True)

    def set_analysis_numbers(self, start, end, skip):
        """Generate subset of image numbers to be analyzed."""
        if self.is_stack:
            npts, *_ = self.stack.shape
            all_nums = list(range(npts))
            nums = all_nums[start:end:skip]
        else:
            files = self.files[start:end:skip]
            nums = [file.num for file in files]
        return nums

    def format_data(self, data):
        """Add file info (name, time, etc.) to analysis results if possible.

        (self.info is defined only if ImgSeries inherits from filo.Series,
        which is not the case if img data is in a stack).
        """
        if self.is_stack:
            self.data = data
        else:
            self.data = pd.concat([self.info, data], axis=1, join='inner')
