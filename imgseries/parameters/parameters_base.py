"""Base classes for display / transform / analysis parameters"""

from filo import ParameterBase, TransformParameterBase, CorrectionParameterBase

from ..config import CONFIG
from ..fileio import FileIO


class Parameter(ParameterBase):
    """Base class to define common methods for different parameters."""

    name = None  # define in subclasses (e.g. "zones")

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            object describing the image series to work with
        """
        super().__init__(data_series=img_series)
        self.img_series = img_series  # ImgSeries object on which to define zones

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
        self.data = all_data[self.name]


class DisplayParameter(Parameter):
    """Base class for global dispaly options (contrast changes, colormaps, etc.)

    These parameters DO NOT impact analysis (only options for image display)
    """

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.img_series.load_display(filename=filename)


class TransformParameter(TransformParameterBase):
    """Base class for global transorms on image series (rotation, crop etc.)

    These parameters DO impact analysis and are stored in metadata.
    """
    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            object describing the image series to work with
        """
        super().__init__(data_series=img_series)
        self.img_series = img_series  # ImgSeries object on which to define zones

    def _load(self, filename=None):
        """Load parameter data from .json file."""
        return self.img_series.load_transforms(filename=filename)


class CorrectionParameter(CorrectionParameterBase):
    """Prameter for corrections (flicker, shaking, etc.) on image series"""

    def __init__(self, img_series):
        """Init parameter object.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            object describing the image series to work with
        """
        super().__init__(data_series=img_series)
        self.img_series = img_series  # ImgSeries object on which to define zones

    def load(self, filename=None):
        """Load parameter data from .json and .tsv files (with same name).

        Redefine Parameter.load() because here stored as tsv file.
        """
        path = self.img_series.savepath
        fname = CONFIG['filenames'][self.name] if filename is None else filename
        try:  # if there is metadata, load it
            filepath = path / (fname + '.json')
        except FileNotFoundError:
            self.metadata = {}
        else:
            self.metadata = FileIO.from_json(filepath)
        self.data = {
            'correction': FileIO.from_tsv(filepath=path / (fname + '.tsv'))
        }


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
        metadata = {self.name: self.data}
        self.analysis.results.save_metadata(metadata, filename=filename)
