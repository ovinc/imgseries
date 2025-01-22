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
    def from_json(path, filename):
        """"Load json file as a dict.

        Parameters
        ----------
        path : pathlib object
            folder containing the file

        filename : str
            name of the file without extension

        Returns
        -------
        dict
        """
        file = path / (filename + '.json')
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def to_json(data, path, filename):
        """"Save data (dict) to json file.

        Parameters
        ----------
        data : dict
            dictionary of data

        path : pathlib object
            folder containing the file

        filename : str
            name of the file without extension

        Returns
        -------
        None
            (writes data to file)
        """
        file = path / (filename + '.json')
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def from_tsv(path, filename):
        """"Load tsv data file as a dataframe.

        Parameters
        ----------
        path : pathlib object
            folder containing the file

        filename : str
            name of the file without extension

        Returns
        -------
        pd.DataFrame
        """
        file = path / (filename + '.tsv')
        return pd.read_csv(file, index_col='num', sep=CONFIG['csv separator'])

    @staticmethod
    def to_tsv(data, path, filename):
        """"Save dataframe to tsv data file.

        Parameters
        ----------
        data : pd.DataFrame
        path : pathlib object
            folder containing the file
        filename : str
            name of the file without extension

        Returns
        -------
        None
            (writes data to file)
        """
        file = path / (filename + '.tsv')
        data.to_csv(file, sep=CONFIG['csv separator'])
