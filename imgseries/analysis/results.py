"""Analysis of image series (base class)"""

# Nonstandard
import gittools
from filo import ResultsBase

# local imports
from ..config import CONFIG
from ..fileio import FileIO


class Results(ResultsBase):
    """Base class for results, can be used as is but won't be able to
    interact with files.
    In order to interact (save/load) with files, define the methods below.
    """

    def _load_data(self, file):
        """Return analysis data from file.

        Parameters
        ----------
        file : pathlib.Path object
            file to load the data from

        Returns
        -------
        Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.
        """
        pass

    def _save_data(self, data, file):
        """Write data to file

        Parameters
        ----------
        data : Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.

        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        pass

    def _load_metadata(self, file):
        """Return analysis metadata from file as a dictionary.

        Parameters
        ----------
        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        dict
            metadata
        """
        pass

    def _save_metadata(self, metadata, file):
        """Write metadata to file

        Parameters
        ----------
        metadata : dict
            Metadata as a dictionary

        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        pass


class PandasTsvJsonResults(Results):
    """"Store data as a pandas dataframe and saves it as a tsv file.

    Metadata is saved as .json
    """

    # define in subclasses (e.g. 'glevel' or 'ctrack')
    measurement_type = None

    # define in subclass (e.g. 'Img_GreyLevel')
    # Note that the program will add .tsv or .json depending on context
    default_filename = 'Results'
    data_extension = '.tsv'
    metadata_extension = '.json'

    def _load_data(self, file):
        """Return analysis data from file.

        Parameters
        ----------
        file : pathlib.Path object
            file to load the data from

        Returns
        -------
        Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.
        """
        return FileIO.from_tsv(file)

    def _save_data(self, data, file):
        """Write data to file

        Parameters
        ----------
        data : Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.

        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        FileIO.to_tsv(data=data, file=file)

    def _load_metadata(self, file):
        """Return analysis metadata from file as a dictionary.

        Parameters
        ----------
        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        dict
            metadata
        """
        return FileIO.from_json(file)

    def _save_metadata(self, metadata, file):
        """Write metadata to file

        Parameters
        ----------
        metadata : dict
            Metadata as a dictionary

        file : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        gittools.save_metadata(
            file=file,
            info=metadata,
            module=CONFIG['checked modules'],
            dirty_warning=True,
            notag_warning=True,
            nogit_ok=True,
            nogit_warning=True,
        )
