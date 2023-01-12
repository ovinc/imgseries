"""General tools for image anlysis."""


# Standard library imports
import json
import os
from pathlib import Path

# Nonstandard
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

# Homemade modules
import filo
import gittools

# local imports
from .config import filenames, csv_separator, checked_modules
from .config import _read, _rgb_to_grey


# ================ General classes for managing image series =================


class ImgSeries(filo.Series):
    """Class to manage series of images, possibly in several folders."""

    # Only for __repr__ (str representation of class object, see filo.Series)
    name = 'Image Series'

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = filenames['files'] + '.tsv'

    def __init__(self, paths='.', extension='.png', savepath='.', stack=None):
        """Init image series object.

        Parameters
        ----------
        - paths can be a string, path object, or a list of str/paths if data
          is stored in multiple folders.
        - savepath: folder in which to save analysis data.
        - extension: extension of files to consider (e.g. '.png')
        - measurement_type: specify 'glevel' or 'ctrack' (optional, for subclasses)

        If file series is in a stack rather than in a series of images:
        - stack: path to the stack (.tiff) file
          (parameters paths & extension will be ignored)
        """
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

    def read(self, num=0):
        """Load image data (image identifier num across folders)."""
        if not self.is_stack:
            return _read(self.files[num].file)
        else:
            return self.stack[num]

    @staticmethod
    def rgb_to_grey(img):
        """"Convert RGB to grayscale"""
        return _rgb_to_grey(img)

    def show(self, num=0, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default for images with 1 color channel)
        """
        img = self.read(num)
        fig, ax = plt.subplots()

        if 'cmap' not in kwargs and img.ndim < 3:
            kwargs['cmap'] = 'gray'

        ax.imshow(img, **kwargs)

        title = 'Image' if self.is_stack else self.files[num].name
        ax.set_title(f'{title} (#{num})')
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
        metadata_file = self.savepath / (name + '.json')
        with open(metadata_file, 'r', encoding='utf8') as f:
            metadata = json.load(f)
        return metadata

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
