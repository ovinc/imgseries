"""Analysis of image series (base class)"""

# Standard library imports
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Nonstandard
import pandas as pd
import gittools
from tqdm import tqdm

# local imports
from .config import filenames, csv_separator, checked_modules, _from_json
from .viewers import ViewerTools


class Analysis(ViewerTools):
    """Base class for analysis subclasses. Used as multiple inheritance."""

    measurement_type = None  # define in subclasses (e.g. 'glevel', 'ctrack', etc.)

    def __init__(self, img_series=None, Viewer=None, savepath=None):
        """Initialize Analysis object

        Parameters
        ----------

        - img_series: image series from the ImgSeries class or subclasses

        - Viewer: which viewer is used to display and inspect analysis data
                  (is used by ViewerTools)

        - savepath: folder in which to save analysis data & metadata
                    (if not specified, the img_series savepath is used)
        """
        self.img_series = img_series
        self.savepath = Path(savepath) if savepath else img_series.savepath

        self.data = None      # Data that will be saved in analysis file

        if self.img_series.is_stack:
            # Below, pre-populate parameters to be saved as metadata.
            # Other metadata will be added to this dict before saving
            # into metadata file
            stack_path = os.path.relpath(self.img_series.stack_path,
                                         self.savepath)
            self.metadata = {'stack': stack_path}
        else:
            folders = [os.path.relpath(f, self.savepath) for f in self.img_series.folders]
            self.metadata = {'path': str(self.savepath.resolve()),
                             'folders': folders}

        ViewerTools.__init__(self, Viewer=Viewer)

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
        self.nums = self.img_series._set_substack(start, end, skip)
        self.nimg = len(self.nums)

        self._initialize()
        self._add_metadata()
        self._prepare_data_storage()

        if parallel:  # ================================= Multiprocessing mode

            futures = {}

            with ProcessPoolExecutor(max_workers=nprocess) as executor:

                for num in self.nums:
                    future = executor.submit(self._analyze, num, live=False)
                    futures[num] = future

                # Waitbar ----------------------------------------------------
                futures_list = list(futures.values())
                for future in tqdm(as_completed(futures_list), total=self.nimg):
                    pass

                # Get results ------------------------------------------------
                for num, future in futures.items():
                    data = future.result()
                    self._store_data(data)

        else:  # ============================================= Sequential mode

            if not live:
                for num in tqdm(self.nums):
                    data = self._analyze(num, live=False)
                    self._store_data(data)
            else:
                # plot uses self.__analyze_live to calculate and store data
                live_plot = self.Viewer(self, live=True)
                # without self.animation, the animation is garbage collected
                self.animation = live_plot.animate(nums=self.nums, blit=blit)

        # Finalize and format data -------------------------------------------

        self._format_data()

    def _analyze_live(self, num):
        data = self._analyze(num, live=True)
        self._store_data(data)
        return data

    def _format_data(self):
        """Add file info (name, time, etc.) to analysis results if possible.

        (self.info is defined only if ImgSeries inherits from filo.Series,
        which is not the case if img data is in a stack).
        """
        data_table = self._generate_pandas_data()

        if self.img_series.is_stack:
            self.data = data_table
        else:
            self.data = pd.concat([self.img_series.info, data_table],
                                  axis=1,
                                  join='inner')

    def _initialize(self):
        """Check everything OK before starting analysis & initialize params.

        Define in subclasses."""
        pass

    def _add_metadata(self):
        """Add useful analysis parameters etc. to the self.metadata dict.

        (later saved in the metadata json file)
        Define in subclasses."""
        pass

    def _prepare_data_storage(self):
        """How to prepare structure(s) that will hold the analyzed data.

        Define in subclasses."""
        pass

    def _analyze(self, num, live=False):
        """Analysis process on single image. Returns data handled by _store_data.

        Parameters
        ----------
        - num: file number identifier across the image file series
        - live: if True, analysis results are displayed in real time


        Output
        ------
        - data, handled by self._store_data()

        Define in subclasses."""

    def _store_data(self, data):
        """How to store data generated by analysis on a single image.

        Define in subclasses."""
        pass

    def _generate_pandas_data(self):
        """How to convert data generated by _store_data() into a pandas table.

        Define in subclasses."""
        pass

    def _set_filename(self, filename):
        return filenames[self.measurement_type] if filename is None else filename

    def _set_substack(self, *args, **kwargs):
        """Needed to be able to use ViewerTools correctly."""
        return self.img_series._set_substack(*args, **kwargs)

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
        name = self._set_filename(filename)
        analysis_file = self.savepath / (name + '.tsv')
        metadata_file = self.savepath / (name + '.json')

        # save analysis data -------------------------------------------------
        self.data.to_csv(analysis_file, sep=csv_separator)

        # save analysis metadata ---------------------------------------------

        self.metadata['rotation'] = self.img_series.rotation.data
        self.metadata['crop'] = self.img_series.crop.data

        gittools.save_metadata(file=metadata_file,
                               info=self.metadata,
                               module=checked_modules,
                               dirty_warning=True,
                               notag_warning=True,
                               nogit_ok=True,
                               nogit_warning=True)

    def load(self, filename=None):
        """Load analysis data from tsv file and return it as pandas DataFrame.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.tsv.
        """
        name = self._set_filename(filename)
        analysis_file = self.savepath / (name + '.tsv')
        data = pd.read_csv(analysis_file, index_col='num', sep=csv_separator)
        return data

    def load_metadata(self, filename=None):
        """Return analysis metadata from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        name = self._set_filename(filename)
        return _from_json(self.savepath, name)

    def regenerate(self, filename=None):
        """Load saved data, metadata and regenerate objects from them.

        Is used to reset the system in a state similar to the end of the
        analysis that was made before saving the results.
        """
        # load data from files
        self.data = self.load(filename=filename)
        self.metadata = self.load_metadata(filename=filename)

        # re-apply transforms (rotation, crop etc.)
        self.img_series.rotation.data = self.metadata['rotation']
        self.img_series.crop.data = self.metadata['crop']
