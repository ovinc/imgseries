"""Base classes for display / transform / analysis parameters"""


import pandas as pd
from ..config import CONFIG, FILENAMES


class Parameter:
    """Base class to define common methods for different parameters."""

    parameter_type = None  # define in subclasses (e.g. "zones")

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. ImgSeries)
        """
        self.img_series = img_series  # ImgSeries object on which to define zones
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def __repr__(self):
        return f'{self.__class__.__name__} object {self.data}'

    def load(self, filename=None):
        """Load parameter data from .json file and put it in self.data.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        self.data = all_data[self.parameter_type]

    def reset(self):
        """Reset parameter data (e.g. rotation angle zero, ROI = total image, etc.)"""
        self.data = {}

    @property
    def is_empty(self):
        return not self.data


class DisplayParameter(Parameter):
    """Base class for global dispaly options (contrast changes, colormaps, etc.)

    These parameters DO NOT impact analysis (only options for image display)
    """

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.img_series.load_display(filename=filename)


class TransformParameter(Parameter):
    """Base class for global transorms on image series (rotation, crop etc.)

    These parameters DO impact analysis and are stored in metadata.
    """

    @property
    def order(self):
        # Order in which transform is applied if several transforms defined
        return self.img_series.transforms.index(self.parameter_type)

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.img_series.load_transforms(filename=filename)

    def reset(self):
        """Reset parameter data (e.g. rotation angle zero, ROI = total image, etc.)"""
        self.data = {}
        self._update_parameters()

    def _clear_cache(self):
        """If images are stored in a cache, clear it so that the new transform
        parameter can be taken into account upon read()"""
        if self.img_series.cache:
            self.img_series.read.cache_clear()

    def _update_parameters(self):
        """What to do when a parameter is updated"""
        self._clear_cache()

        try:
            subtraction = self.img_series.subtraction
        except AttributeError:
            return

        if not subtraction.is_empty and self.order <= subtraction.order:
            subtraction._update_reference_image()


class CorrectionParameter(Parameter):
    """Prameter for corrections (flicker, shaking, etc.) on image series"""

    parameter_type = 'flicker'

    def load(self, filename=None):
        """Load parameter data from .json and .tsv files (with same name).

        Redefine Parameter.load() because here stored as tsv file.
        """
        path = self.img_series.savepath
        fname = CONFIG['filenames'][self.parameter_type] if filename is None else filename
        try:  # if there is metadata, load it
            self.data = self.img_series.file_manager.from_json(path, fname)
        except FileNotFoundError:
            self.data = {}
        self.data['correction'] = self.img_series.file_manager.from_tsv(path, fname)


class AnalysisParameter(Parameter):
    """Base class for parameters used in analysis (contours, zones, etc.)"""

    def __init__(self, analysis):
        """Init parameter object.

        Parameters
        ----------
        - analysis: object of an analysis class (e.g. GreyLevel)
        """
        self.analysis = analysis  # Analysis object (grey level)
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.analysis.results._load_metadata(filename=filename)

    def save(self, filename=None):
        """Save info about parameter in json file."""
        metadata = {self.parameter_type: self.data}
        self.analysis.results._save_metadata(metadata, filename=filename)
