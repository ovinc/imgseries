"""Analysis of image series (base class)"""


from filo import ResultsBase

# local imports
from ..config import CONFIG
from ..fileio import FileIO


class PandasTsvJsonResults(ResultsBase):
    """"Store data as a pandas dataframe and saves it as a tsv file.

    Metadata is saved as .json
    """

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
        return FileIO.to_json_with_gitinfo(data=metadata, filepath=filepath)
