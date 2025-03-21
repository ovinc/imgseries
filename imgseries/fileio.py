"""Image / File managers"""


# Nonstandard
import pandas as pd
import skimage
from filo import to_json, load_json

# local imports
from .config import CONFIG


class FileIO:

    @staticmethod
    def read_single_image(filepath):
        """Read a single image from an image file

        Parameters
        ----------
        filepath : str or Path object

        Returns
        -------
        array_like
            image as an array
        """
        return skimage.io.imread(filepath)

    @staticmethod
    def read_tiff_stack_whole(filepath):
        """load file into image array (file: pathlib Path object).

        Parameters
        ----------
        filepath : str or Path object

        Returns
        -------
        array_like
            image stack as an array
        """
        return skimage.io.imread(filepath)

    @staticmethod
    def read_tiff_stack_slice(filepath, num):
        """load file into image array (file: pathlib Path object).

        Parameters
        ----------
        filepath : str or Path object

        Returns
        -------
        array_like
            image as an array
        """
        return skimage.io.imread(filepath, key=num)

    @staticmethod
    def from_json(filepath):
        """"Load json file as a dict.

        Parameters
        ----------
        filepath : pathlib object
            file to load the data from

        Returns
        -------
        dict
        """
        return load_json(filepath=filepath)

    @staticmethod
    def to_json(data, filepath):
        """"Save data (dict) to json file.

        Parameters
        ----------
        data : dict
            dictionary of data

        filepath : pathlib object
            file to write the data into

        Returns
        -------
        None
            (writes data to file)
        """
        return to_json(data=data, filepath=filepath)

    @staticmethod
    def from_tsv(filepath):
        """"Load tsv data file as a dataframe.

        Parameters
        ----------
        filepath : pathlib object
            file to read the data from

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_csv(filepath, index_col='num', sep=CONFIG['csv separator'])

    @staticmethod
    def to_tsv(data, filepath):
        """"Save dataframe to tsv data file.

        Parameters
        ----------
        data : pd.DataFrame
        filepath : pathlib object
            file to write the data into

        Returns
        -------
        None
            (writes data to file)
        """
        data.to_csv(filepath, sep=CONFIG['csv separator'])
