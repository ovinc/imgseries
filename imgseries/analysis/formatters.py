"""Base classes for Formatters for analysis of image series"""

# Standard
import os
from abc import abstractmethod

# Nonstandard
import pandas as pd
from filo import FormatterBase, PandasFormatterBase


def _get_path_metadata(analysis):
    """Return dict about image path(s)."""
    savepath = analysis.results.savepath
    if analysis.img_series.is_stack:
        stack_path = os.path.relpath(
            path=analysis.img_series.path,
            start=savepath,
        )
        return {'stack': stack_path}
    else:
        folders = [
            os.path.relpath(path=f, start=savepath)
            for f in analysis.img_series.files.folders
        ]
        return {'path': str(savepath.resolve()), 'folders': folders}


class Formatter(FormatterBase):
    """Base class for formatting of results spit out by analysis methods"""

    # ================ Redefinition of FormatterBase methods =================

    def _to_results_metadata(self):
        metadata = super()._to_results_metadata()
        path_metadata = _get_path_metadata(self.analysis)
        return {**path_metadata, **metadata}

    # ================= Subclassing of FormatterBase methods =================

    def _regenerate_additional_data(self, num):
        """How to go back to raw data from data stored in results"""
        return {'image': self.analysis.img_series.read(num=num)}

    # ================== FormatterBase methods to subclass ===================

    @abstractmethod
    def _prepare_data_storage(self):
        """Prepare structure(s) that will hold the analyzed data"""
        pass

    @abstractmethod
    def _store_data(self, data):
        """How to store data generated by analysis on a single image.

        Input
        -----
        data : dict
            Dictionary of data, output of Analysis.analyze()
        """
        pass

    @abstractmethod
    def _to_results_data(self):
        """How to pass stored data into a Results class/subclass.

        (is executed at the end of analysis)

        Returns
        ------
        Any
            data in the format that will be stored in results.data
        """
        return None

    def _to_metadata(self):
        """What metadata to save in the Results class/subclass.

        [OPTIONAL]
        (is executed at the end of analysis)

        Returns
        ------
        dict
            metadata dictionary that will be stored in results.metadata
            (note: transform metadata added automatically)
        """
        return {}

    def _regenerate_analysis_data(self, num):
        """How to go back to raw data (as spit out by the analysis methods
        during analysis) from data saved in results or files.

        [OPTIONAL]

        Useful for plotting / animating results again after analysis, among
        other things.

        Parameters
        ----------
        num : int
            data identifier in the data series

        Returns
        -------
        dict
            data in the format generated by analysis.analyze()

        Notes
        -----
            'num' key is added automatically by _regenerate_data_from_results()
            in the output dict.
        """
        return {}


class PandasFormatter(PandasFormatterBase):
    """Base class for formatting results as a pandas DataFrame"""

    # ================ Redefinition of FormatterBase methods =================

    def _to_results_metadata(self):
        metadata = super()._to_results_metadata()
        path_metadata = _get_path_metadata(self.analysis)
        return {**path_metadata, **metadata}

    # ================= Subclassing of FormatterBase methods =================

    def _regenerate_additional_data(self, num):
        """How to go back to raw data from data stored in results"""
        return {'image': self.analysis.img_series.read(num=num)}

    def _to_results_data(self):
        """Add file info (name, time, etc.) to analysis results if possible.

        (img_series.info is defined only if ImgSeries inherits from filo.Series,
        which is not the case if img data is in a stack).
        """
        data = self.data.sort_index()
        if self.analysis.img_series.is_stack:
            return data
        else:
            info = self.analysis.img_series.info
            return pd.concat([info, data], axis=1, join='inner')

    # ================== FormatterBase methods to subclass ===================

    def _to_metadata(self):
        """What metadata to save in the Results class/subclass.

        [OPTIONAL]
        (is executed at the end of analysis)

        Returns
        ------
        dict
            metadata dictionary that will be stored in results.metadata
            (note: transform metadata added automatically)
        """
        return {}

    # =============== PandasFormatterBase methods to subclass ================

    @abstractmethod
    def _column_names(self):
        """Columns of the analysis data (iterable)"""
        pass

    @abstractmethod
    def _data_to_results_row(self, data):
        """Generate iterable of data that fits in the defined columns.

        Input
        -----
        data is a dictionary, output of Analysis.analyze()

        Returns
        -------
        iterable
            must have a length equal to self._column_names()
        """
        pass

    def _results_row_to_data(self, row):
        """Go from row of data to raw data

        [Optional]
        """
        pass
