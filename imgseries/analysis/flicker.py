"""Grey Level analysis, applied to correction of flicker in image series."""


# Nonstandard imports
import numpy as np

# Local imports
from .formatters import PandasFormatter
from .results import PandasTsvJsonResults
from .grey_level import GreyLevel, GreyLevelViewer
from .grey_level import GreyLevelFormatter


# ============================ Results formatting ============================


class FlickerFormatter(PandasFormatter):

    # If results are independent (results from one num do not depend from
    # analysis on other nums), one do not need to re-do the analysis when
    # asking for the same num twice.
    independent_results = True

    def _column_names(self):
        """Prepare structure(s) that will hold the analyzed data."""
        return list(self.analysis.zones.data) + ['ratio']

    def _data_to_results_row(self, data):
        """Generate iterable of data that fits in the defined columns."""
        # Note that glevels in this case is actually already a ratio
        # (it is the ratio for every defined zone, while 'ratio' is the average)
        # This is a trick to take advantage of as many things defined in
        # GreyLevel as possible.
        return data['glevels'] + [data['ratio']]

    def _results_row_to_data(self, row):
        """Go from row of data to raw data"""
        return {'glevels': list(row.iloc[:-1]), 'ratio': row.iloc[-1]}

    def _to_metadata(self):
        """Get analysis metadata excluding paths and transforms"""
        info = GreyLevelFormatter._to_metadata(self)
        info['reference'] = self.analysis.reference
        return info


class FlickerResults(PandasTsvJsonResults):
    default_filename = 'Img_Flicker'


# ======================== Plotting / Animation class ========================


class FlickerViewer(GreyLevelViewer):
    pass


# =========================== Main ANALYSIS class ============================


class Flicker(GreyLevel):
    """Class to perform analysis of flicker on image series

    (based on Grey Level analysis)

    Class attributes
    ----------------
    Viewer : class
        (subclass of AnalysisViewer)
        Viewer class/subclasses that is used to display and inspect
        analysis data (is used by ViewerTools)

    Formatter: class
        (subclass of Formatter)
        class used to format results spit out by the raw analysis into
        something storable/saveable by the Results class.

    Results : class
        (subclass of Results)
        Results class/subclasses that is used to store, save and load
        analysis data and metadata.
    """
    Viewer = FlickerViewer
    Formatter = FlickerFormatter
    Results = FlickerResults

    # If results are independent (results from one num do not depend from
    # analysis on other nums), one do not need to re-do the analysis when
    # asking for the same num twice, and parallel computing is possible
    independent_results = True

    def __init__(
        self,
        img_series,
        savepath=None,
        reference=0,
        func=np.mean,
    ):
        """Analysis of flicker on selected zone in series of images.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            image series on which the analysis will be run

        savepath : str or Path object
            folder in which to save analysis data & metadata
                    (if not specified, the img_series savepath is used)

        reference : int
            num of image which will serve to normalize others.

        func : function
            function to be applied on the image pixels in the defined
            analysis zones (default: np.mean). Other typical functions
            can be: np.sum, np.max, etc.
        """
        super().__init__(
            img_series=img_series,
            savepath=savepath,
            func=func,
        )
        self.reference = reference

    def _init_analysis(self):
        """Check everything OK before starting analysis & initialize params."""
        super()._init_analysis()

        # I have to do it this way because due to inheritance properties
        # calling super() or GreyLevel() analyze() will call the
        # Flicker _analyze() and not the GreyLevel one.
        img = self.img_series.read(num=self.reference)
        data = super()._analyze(img=img)
        self.ref_values = data['glevels']

    def _analyze(self, img):
        """Basic analysis function, to be threaded or multiprocessed.

        Parameters
        ----------
        img : array_like
            image array to be analyzed (e.g. numpy array).

        Returns
        -------
        dict
            dict of data, handled by self.formatter._store_data()
        """
        data = super()._analyze(img)
        ratios = [
            glevel / ref_val
            for glevel, ref_val in zip(data['glevels'], self.ref_values)
        ]
        # Note that glevels in this case is actually already a ratio
        # (it is the ratio for every defined zone, while 'ratio' is the average)
        # This is a trick to take advantage of as many things defined in
        # GreyLevel as possible.
        data['glevels'] = ratios
        data['ratio'] = np.mean(ratios)
        return data
