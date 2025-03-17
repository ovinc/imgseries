"""Base classes for display / transform / analysis parameters"""

from ..config import CONFIG
from ..fileio import FileIO


class Parameter:
    """Base class to define common methods for different parameters."""

    parameter_name = None  # define in subclasses (e.g. "zones")

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            object describing the image series to work with
        """
        self.img_series = img_series  # ImgSeries object on which to define zones
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def __repr__(self):
        return f'{self.__class__.__name__} object {self.data}'

    def load(self, filename=None):
        """Load parameter data from .json file and put it in self.data.

        Parameters
        ----------
        filename : str

            If filename is not specified, use default filenames.

            If filename is specified, it must be an str without the extension, e.g.
            filename='Test' will load from Test.json.
        """
        self.reset()  # useful when using caching
        all_data = self._load(filename=filename)
        self.data = all_data[self.parameter_name]

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
        return self.img_series.transforms.index(self.parameter_name)

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

    def apply(self, img):
        """How to apply the transform on an image array

        To be defined in subclasses.

        Parameters
        ----------
        img : array_like
            input image on which to apply the transform

        Returns
        -------
        array_like
            the processed image
        """
        pass


class CorrectionParameter(Parameter):
    """Prameter for corrections (flicker, shaking, etc.) on image series"""

    parameter_name = 'flicker'

    def load(self, filename=None):
        """Load parameter data from .json and .tsv files (with same name).

        Redefine Parameter.load() because here stored as tsv file.
        """
        path = self.img_series.savepath
        fname = CONFIG['filenames'][self.parameter_name] if filename is None else filename
        try:  # if there is metadata, load it
            file = path / (fname + '.json')
            self.data = FileIO.from_json(file)
        except FileNotFoundError:
            self.data = {}
        self.data['correction'] = FileIO.from_tsv(file=path + (fname + '.tsv'))


class AnalysisParameter(Parameter):
    """Base class for parameters used in analysis (contours, zones, etc.)"""

    def __init__(self, analysis):
        """Init parameter object.

        Parameters
        ----------
        analysis :  Analysis object
            object of an analysis class (e.g. GreyLevel)
        """
        self.analysis = analysis  # Analysis object (grey level)
        self.data = {}  # dict, e.g. {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.analysis.results.load_metadata(filename=filename)

    def save(self, filename=None):
        """Save info about parameter in json file."""
        metadata = {self.parameter_name: self.data}
        self.analysis.results.save_metadata(metadata, filename=filename)
