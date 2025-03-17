"""Image / File managers"""

# Standard library imports
import json

# Nonstandard
import pandas as pd
import skimage

# local imports
from .config import CONFIG


class FileIO:

    @staticmethod
    def read_single_image(file):
        """Read a single image from an image file

        Parameters
        ----------
        file : str or Path object

        Returns
        -------
        array_like
            image as an array
        """
        return skimage.io.imread(file)

    @staticmethod
    def read_tiff_stack_whole(file):
        """load file into image array (file: pathlib Path object).

        Parameters
        ----------
        file : str or Path object

        Returns
        -------
        array_like
            image stack as an array
        """
        return skimage.io.imread(file)

    @staticmethod
    def read_tiff_stack_slice(file, num):
        """load file into image array (file: pathlib Path object).

        Parameters
        ----------
        file : str or Path object

        Returns
        -------
        array_like
            image as an array
        """
        return skimage.io.imread(file, key=num)

    @staticmethod
    def from_json(file):
        """"Load json file as a dict.

        Parameters
        ----------
        file : pathlib object
            file to load the data from

        Returns
        -------
        dict
        """
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def to_json(data, file):
        """"Save data (dict) to json file.

        Parameters
        ----------
        data : dict
            dictionary of data

        file : pathlib object
            file to write the data into

        Returns
        -------
        None
            (writes data to file)
        """
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def from_tsv(file):
        """"Load tsv data file as a dataframe.

        Parameters
        ----------
        file : pathlib object
            file to read the data from

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_csv(file, index_col='num', sep=CONFIG['csv separator'])

    @staticmethod
    def to_tsv(data, file):
        """"Save dataframe to tsv data file.

        Parameters
        ----------
        data : pd.DataFrame
        file : pathlib object
            file to write the data into

        Returns
        -------
        None
            (writes data to file)
        """
        data.to_csv(file, sep=CONFIG['csv separator'])
