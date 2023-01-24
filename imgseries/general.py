"""General tools for image anlysis."""


# Standard library imports
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Nonstandard
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from skimage import io
import filo
import gittools
from tqdm import tqdm

# local imports
from .config import filenames, csv_separator, checked_modules
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

        ax.imshow(img, **kwargs)

        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.04])
        slider = Slider(ax=ax_slider,
                        label='#',
                        valmin=0,
                        valinit=0,
                        valmax=self.total_img_number - 1,
                        valstep=1,
                        color='steelblue', alpha=0.3)

        def update(num):
            img = self.read(num)
            ax.clear()
            ax.imshow(img, **kwargs)

        slider.on_changed(update)

        return slider


class Analysis:
    """Tools for analysis subclasses. Used as multiple inheritance."""

    def __init__(self, measurement_type=None):
        """Parameters:

        - measurement_type: specify 'glevel', 'ctrack',

        NOTE: subclasses must define a self.plot_type object, which indicates
        which plotting class has to be used for live visualization.
        (ONLY if live view is needed).
        """
        self.measurement_type = measurement_type  # for data loading/saving
        self.data = None      # Data that will be saved in analysis file

        if self.is_stack:
            # Below, pre-populate parameters to be saved as metadata.
            # Other metadata will be added to this dict before saving
            # into metadata file
            stack_path = os.path.relpath(self.stack_path, self.savepath)
            self.metadata = {'stack': stack_path}
        else:
            folders = [os.path.relpath(f, self.savepath) for f in self.folders]
            self.metadata = {'path': str(self.savepath.resolve()),
                             'folders': folders}

    def run(self,
            start=0, end=None, skip=1,
            parallel=False, nprocess=None,
            live=False, blit=False):
        """Start analysis of image sequence.

        PARAMETERS
        ----------
        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - parallel: if True, distribute computation across different processes.
          (only available if calculations on each image is independent of
          calculations on the other images)

        - nprocess: number of process workers; if None (default), use default
          in ProcessPoolExecutor, depends on the number of cores of computer)

        - live: if True, plot analysis results in real time.
        - blit: if True, use blitting to speed up live display

        OUTPUT
        ------
        Pandas dataframe with image numbers as the index, and with columns
        containing timestamps and the calculated data.

        WARNING
        -------
        If running on a Windows machine and using the parallel option, the
        function call must not be run during import of the file containing
        the script (i.e. the function must be in a `if __name__ == '__main__'`
        block). This is because apparently multiprocessing imports the main
        program initially, which causes recursive problems.
        """
        self.nums = self.set_analysis_numbers(start, end, skip)
        self.nimg = len(self.nums)

        self.initialize()
        self.add_metadata()
        self.prepare_data_storage()

        if parallel:  # ================================= Multiprocessing mode

            futures = {}

            with ProcessPoolExecutor(max_workers=nprocess) as executor:

                for num in self.nums:
                    future = executor.submit(self.analysis, num, live)
                    futures[num] = future

                # Waitbar ----------------------------------------------------
                futures_list = list(futures.values())
                for future in tqdm(as_completed(futures_list), total=self.nimg):
                    pass

                # Get results ------------------------------------------------
                for num, future in futures.items():
                    data = future.result()
                    self.store_data(data)

        else:  # ============================================= Sequential mode

            if not live:
                for num in tqdm(self.nums):
                    data = self.analyze(num, live)
                    self.store_data(data)
            else:
                # plot uses self.live_analysis to calculate and store data
                self.live_plot = self.plot_type(analysis=self, blit=blit)

        # Finalize and format data -------------------------------------------

        self.format_data()

    def live_analysis(self, num):
        data = self.analyze(num, live=True)
        self.store_data(data)
        return data

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

    def format_data(self):
        """Add file info (name, time, etc.) to analysis results if possible.

        (self.info is defined only if ImgSeries inherits from filo.Series,
        which is not the case if img data is in a stack).
        """
        data_table = self.generate_pandas_data()

        if self.is_stack:
            self.data = data_table
        else:
            self.data = pd.concat([self.info, data_table],
                                  axis=1,
                                  join='inner')

    def initialize(self):
        """Check everything OK before starting analysis & initialize params.

        Define in subclasses."""
        pass

    def add_metadata(self):
        """Add useful analysis parameters etc. to the self.metadata dict.

        (later saved in the metadata json file)
        Define in subclasses."""
        pass

    def prepare_data_storage(self):
        """How to prepare structure(s) that will hold the analyzed data.

        Define in subclasses."""
        pass

    def analyze(self, num, live=False):
        """Analysis process on single image. Returns data handled by store_data.

        Parameters
        ----------
        - num: file number identifier across the image file series
        - live: if True, analysis results are displayed in real time


        Output
        ------
        - data, handled by self.store_data()

        Define in subclasses."""

    def store_data(self, data):
        """How to store data generated by analysis on a single image.

        Define in subclasses."""
        pass

    def generate_pandas_data(self):
        """How to convert data generated by store_data() into a pandas table.

        Define in subclasses."""
        pass

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

        self.metadata['rotation'] = self.rotation.data
        self.metadata['crop'] = self.crop.data

        gittools.save_metadata(file=metadata_file, info=self.metadata,
                               module=checked_modules,
                               dirty_warning=True, notag_warning=True,
                               nogit_ok=True, nogit_warning=True)

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
