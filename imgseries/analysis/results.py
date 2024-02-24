"""Analysis of image series (base class)"""

# Standard library imports
from pathlib import Path

# Nonstandard
import gittools

# local imports
from ..config import CONFIG
from ..managers import FileManager


class Results:
    """Base class for classes that stores results and interacts with files

    Metadata is automatically stored in a JSON file with the same name as the
    analysis results file.
    """

    # define in subclasses (e.g. 'glevel' or 'ctrack')
    measurement_type = None

    # define in subclass (e.g. 'Img_GreyLevel')
    # Note that the program will add .tsv or .json depending on context
    default_filename = 'Results'

    def __init__(self, savepath='.'):
        self.reset()  # creates self.data and self.metadata
        self.savepath = Path(savepath)

    def _set_filename(self, filename):
        return self.default_filename if filename is None else filename

    def reset(self):
        """Erase data and metadata from the results."""
        self.data = None
        self.metadata = {}

    # =============== Interacting with files (saving/loading) ================

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
        self._save_data()
        self._save_metadata(metadata=self.metadata)

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

    def _load_metadata(self, filename=None):
        """Return analysis metadata from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        name = self._set_filename(filename)
        return FileManager.from_json(self.savepath, name)

    def _save_metadata(self, metadata, filename=None):
        """Inverse of _load_metadata"""
        name = self._set_filename(filename)
        metadata_file = self.savepath / (name + '.json')

        gittools.save_metadata(
            file=metadata_file,
            info=metadata,
            module=CONFIG['checked modules'],
            dirty_warning=True,
            notag_warning=True,
            nogit_ok=True,
            nogit_warning=True,
        )

    # ----------- To define in subclasses (how to save/load data) ------------

    def _load_data(self, filename=None):
        """Load analysis data from tsv file and return it as pandas DataFrame.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.tsv.
        """
        pass

    def _save_data(self, filename=None):
        """Inverse of _load_data()"""
        pass


class PandasTsvResults(Results):
    """"Store data as a pandas dataframe and saves it as a tsv file.

    Metadata is saved as .Json (see Results)
    """

    def _load_data(self, filename=None):
        """Load analysis data from tsv file and return it as pandas DataFrame.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.tsv.
        """
        return FileManager.from_tsv(
            path=self.savepath,
            filename=self._set_filename(filename)
        )

    def _save_data(self, filename=None):
        """Inverse of _load_data()"""
        FileManager.to_tsv(
            data=self.data,
            path=self.savepath,
            filename=self._set_filename(filename),
        )
