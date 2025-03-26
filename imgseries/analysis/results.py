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
    In order to interact (save/load) with files, define the following methods:
    - _load_data()
    - _save_data()
    - _load_metadata()
    - _save_metadata()
    """
    pass


class PandasTsvJsonResults(Results):
    """"Store data as a pandas dataframe and saves it as a tsv file.

    Metadata is saved as .json
    """
    # define in subclass (e.g. 'Img_GreyLevel')
    # Note that the program will add .tsv or .json depending on context
    default_filename = 'Results'
    data_extension = '.tsv'
    metadata_extension = '.json'

    def _load_data(self, filepath):
        """Return analysis data from file.

        Parameters
        ----------
        filepath : pathlib.Path object
            file to load the data from

        Returns
        -------
        Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.
        """
        return FileIO.from_tsv(filepath)

    def _save_data(self, data, filepath):
        """Write data to file

        Parameters
        ----------
        data : Any
            Data in the form specified by user in _load_data()
            Typically a pandas dataframe.

        filepath : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        FileIO.to_tsv(data=data, filepath=filepath)

    def _load_metadata(self, filepath):
        """Return analysis metadata from file as a dictionary.

        Parameters
        ----------
        filepath : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        dict
            metadata
        """
        return FileIO.from_json(filepath)

    def _save_metadata(self, metadata, filepath):
        """Write metadata to file

        Parameters
        ----------
        metadata : dict
            Metadata as a dictionary

        filepath : pathlib.Path object
            file to load the metadata from

        Returns
        -------
        None
        """
        gittools.save_metadata(
            file=filepath,
            info=metadata,
            module=CONFIG['checked modules'],
            dirty_warning=True,
            notag_warning=True,
            nogit_ok=True,
            nogit_warning=True,
        )
