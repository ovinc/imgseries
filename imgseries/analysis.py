"""Analysis of image series (base class)"""

# Standard library imports
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Nonstandard
import pandas as pd
import gittools
from tqdm import tqdm

# local imports
from .config import filenames, csv_separator, checked_modules


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
        self.nums = self._set_substack(start, end, skip)
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
