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
from ..config import CONFIG
from ..managers import FileManager
from ..viewers import ViewerTools


class Analysis(ViewerTools):
    """Base class for analysis subclasses (GreyLevel, ContourTracking, etc.)."""

    measurement_type = None  # define in subclasses (e.g. 'glevel', 'ctrack', etc.)

    DefaultViewer = None     # define in subclasses
    DefaultFormatter = None  # define in subclasses
    DefaultResults = None    # define in subclasses

    def __init__(self,
                 img_series,
                 savepath=None,
                 Viewer=None,
                 Formatter=None,
                 Results=None,
                 ):
        """Initialize Analysis object

        Parameters
        ----------

        - img_series: image series from the ImgSeries class or subclasses

        - savepath: folder in which to save analysis data & metadata
                    (if not specified, the img_series savepath is used)

        - Viewer: Viewer class/subclasses that is used to display and inspect
                  analysis data (is used by ViewerTools)

        - Formatter: class/subclass of Formatter to format results spit out
                     by the raw analysis into something storable/saveable
                     by the Results class.

        - Results: Results class/subclasses that is used to store, save and
                   load analysis data and metadata.
        """
        Viewer = self.DefaultViewer if Viewer is None else Viewer
        Formatter = self.DefaultFormatter if Formatter is None else Formatter
        Results = self.DefaultResults if Results is None else Results

        self.img_series = img_series
        self.Viewer = Viewer
        self.formatter = Formatter(self)

        savepath = Path(savepath) if savepath else img_series.savepath
        self.results = Results(savepath=savepath)

        if self.img_series.is_stack:
            # Below, pre-populate parameters to be saved as metadata.
            # Other metadata will be added to this dict before saving
            # into metadata file
            stack_path = os.path.relpath(self.img_series.stack_path, savepath)
            self.results.metadata['stack'] = stack_path
        else:
            folders = [os.path.relpath(f, savepath) for f in self.img_series.folders]
            self.results.metadata['path'] = str(savepath.resolve()),
            self.results.metadata['folders'] = folders

        ViewerTools.__init__(self, Viewer=Viewer)

    # ========================= Misc. internal tools =========================

    def _set_substack(self, *args, **kwargs):
        """Needed to be able to use ViewerTools correctly."""
        return self.img_series._set_substack(*args, **kwargs)

    def _add_transform_to_metadata(self):
        """Add information about image transforms (rotation, crop etc.) to metadata."""
        for transform_name in CONFIG['image transforms']:
            transform_data = getattr(self.img_series, transform_name).data
            self.results.metadata[transform_name] = transform_data

    def _analyze_live(self, num):
        data = self._analyze(num, live=True)
        self.formatter._store_data(data)
        return data

    # ============================ Public methods ============================

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
        self._add_transform_to_metadata()

        self.formatter._prepare_data_storage()

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
                    self.formatter._store_data(data)

        else:  # ============================================= Sequential mode

            if not live:
                for num in tqdm(self.nums):
                    data = self._analyze(num, live=False)
                    self.formatter._store_data(data)
            else:
                # plot uses self.__analyze_live to calculate and store data
                live_plot = self.Viewer(self, live=True)
                # without self.animation, the animation is garbage collected
                self.animation = live_plot.animate(nums=self.nums, blit=blit)

        # Finalize and format data -------------------------------------------

        self.formatter._save_results()

    def regenerate(self, filename=None):
        """Load saved data, metadata and regenerate objects from them.

        Is used to reset the system in a state similar to the end of the
        analysis that was made before saving the results.
        """
        # load data from files
        self.results.load(filename=filename)

        # re-apply transforms (rotation, crop etc.)

        for transform_name in CONFIG['image transforms']:

            # e.g. self.img_series.crop.reset()
            getattr(self.img_series, transform_name).reset()

            # e.g. self.img_series.crop.data = self.results.metadata['crop']
            data = self.results.metadata.get(transform_name, {})
            setattr(getattr(self.img_series, transform_name), 'data', data)

    # =================== Methods to define in subclasses ====================

    def _initialize(self):
        """Check everything OK before starting analysis & initialize params.

        Define in subclasses."""
        pass

    def _add_metadata(self):
        """Add useful analysis parameters etc. to the self.metadata dict.

        (later saved in the metadata json file)
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
        pass


# ================================ Formatters ================================


class Formatter:
    """Base class for formatting of results spit out by analysis methods"""

    def __init__(self, analysis):
        """Link formatter to analysis object"""
        self.analysis = analysis

    def _prepare_data_storage(self):
        """Prepare structure(s) that will hold the analyzed data (raw)."""
        pass

    def _store_data(self, data):
        """How to store data generated by analysis on a single image."""
        pass

    def _save_results(self):
        """How to pass stored data into an AnlysisResults class/subclass."""
        pass

    def _regenerate_data(self, num):
        """How to go back to raw data (as spit out by the analysis methods
        during analysis) from data saved in results or files.

        Useful for plotting / animating results again after analysis, among
        other things.
        """
        pass


class PandasFormatter(Formatter):
    """Base class for formatting of results ad a pandas dataframe"""

    def _generate_pandas_data(self):
        """How to convert data generated by _store_data() into a pandas table.

        Define in subclass.
        """
        pass

    def _save_results(self):
        """Add file info (name, time, etc.) to analysis results if possible.

        (img_series.info is defined only if ImgSeries inherits from filo.Series,
        which is not the case if img data is in a stack).
        """
        data_table = self._generate_pandas_data()

        if self.analysis.img_series.is_stack:
            self.analysis.results.data = data_table
        else:
            info = self.analysis.img_series.info
            self.analysis.results.data = pd.concat([info, data_table],
                                                   axis=1,
                                                   join='inner')


# ================================= Results ==================================


class Results:
    """Base class for classes that stores and loads analysis results."""

    measurement_type = None  # define in subclasses (e.g. 'glevel' or 'ctrack')

    def __init__(self, savepath='.'):
        self.reset()
        self.savepath = Path(savepath)

    def _set_filename(self, filename):
        return CONFIG['filenames'][self.measurement_type] if filename is None else filename

    def reset(self):
        """Erase data and metadata from the results."""
        self.data = None
        self.metadata = {}

    def save(self, filename=None):
        """Save analysis data and metadata into .tsv / .json files.

        Define in subclass."""
        pass

    def load(self, filename=None):
        """Load analysis data and metadata and stores it in self.data/metadata.

        Define in subclass.
        """
        pass


class PandasTsvResults(Results):
    """"Store data as a pandas dataframe and saves it as a tsv file.

    Metadata is saved as .Json
    """

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
        self.data.to_csv(analysis_file, sep=CONFIG['csv separator'])

        # save analysis metadata ---------------------------------------------

        gittools.save_metadata(file=metadata_file,
                               info=self.metadata,
                               module=CONFIG['checked modules'],
                               dirty_warning=True,
                               notag_warning=True,
                               nogit_ok=True,
                               nogit_warning=True)

    def load(self, filename=None):
        """Load analysis data and metadata and stores it in self.data/metadata.
        Parameters
        ----------
        filename:

        - If filename is not specified, use default filenames.

        - If filename is specified, it must be an str without the extension
            e.g. filename='Test' will create Test.tsv and Test.json files,
            containing tab-separated data file and metadata file, respectively.
        """
        self.data = self._load_data(filename=filename)
        self.metadata = self._load_metadata(filename=filename)

    def _load_data(self, filename=None):
        """Load analysis data from tsv file and return it as pandas DataFrame.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.tsv.
        """
        name = self._set_filename(filename)
        analysis_file = self.savepath / (name + '.tsv')
        data = pd.read_csv(analysis_file, index_col='num', sep=CONFIG['csv separator'])
        return data

    def _load_metadata(self, filename=None):
        """Return analysis metadata from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        name = self._set_filename(filename)
        return FileManager.from_json(self.savepath, name)