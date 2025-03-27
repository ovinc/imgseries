"""Analysis of wetting / drying fronts."""

# Non-standard modules
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .analysis_base import Analysis
from .formatters import PandasFormatter
from .results import PandasTsvJsonResults
from ..viewers import AnalysisViewer


# ============================ Results formatting ============================


class Front1DFormatter(PandasFormatter):

    def _column_names(self):
        """Prepare structure(s) that will hold the analyzed data."""
        return range(self.analysis.img_series.nx)

    def _data_to_results_row(self, data):
        """Generate iterable of data that fits in the defined columns."""
        return data['analysis']

    def _results_row_to_data(self, row):
        """Go from row of data to raw data"""
        return {'analysis': row.values}

    def _to_metadata(self):
        """Get analysis metadata excluding paths and transforms"""
        return {}


class Front1DResults(PandasTsvJsonResults):

    default_filename = 'Img_Front1D'

    @staticmethod
    def _integrify_columns(name):
        """Function used to make column names integers when needed."""
        try:
            return int(name)
        except ValueError:
            return name

    def _load_data(self, filepath):
        """load_csv returns the indices as str, but we need to convert the
        x positions as intergers"""
        data = super()._load_data(filepath)
        return data.rename(columns=self._integrify_columns)


# ======================== Plotting / Animation class ========================


class Front1DViewer(AnalysisViewer):

    # ---------------- Methods subclassed from AnalysisViewer ----------------

    def _create_figure(self):
        self.fig = plt.figure(figsize=(5, 7))
        xmin = 0.05
        xmax = 0.95
        w = xmax - xmin
        self.ax_img = self.fig.add_axes([xmin, 0.33, w, 0.65])
        self.ax_analysis = self.fig.add_axes([xmin, 0.09, w, 0.25])
        self.axs = self.ax_img, self.ax_analysis
        self.ax_img.axis('off')

    def _first_plot(self, data):
        """What to do the first time data arrives on the plot."""
        self._create_image(data)

        analysis_data = data.get('analysis', None)
        if analysis_data is None:  # e.g. start num not analyzed
            nx = self.analysis.img_series.nx
            self.analysis_line, = self.ax_analysis.plot(np.zeros(nx))
            self.analysis_line.set_visible(False)
        else:
            self.analysis_line, = self.ax_analysis.plot(analysis_data)

        self.updated_artists = [self.analysis_line, self.imshow]

    def _update_plot(self, data):
        """What to do upon iterations of the plot after the first time."""
        self._update_image(data)

        if data.get('analysis', None) is None:  # e.g. start num not analyzed
            self.analysis_line.set_visible(False)
            return

        self.analysis_line.set_visible(True)
        self.analysis_line.set_ydata(data['analysis'])
        self._autoscale(ax=self.ax_analysis)


# =========================== Main ANALYSIS class ============================


class Front1D(Analysis):
    """Class to perform analysis 1D fronts on image series.

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
    Viewer = Front1DViewer
    Formatter = Front1DFormatter
    Results = Front1DResults

    # If results are independent (results from one num do not depend from
    # analysis on other nums), one do not need to re-do the analysis when
    # asking for the same num twice, and parallel computing is possible
    independent_results = True

    def __init__(
        self,
        img_series,
        savepath=None,
    ):
        """Analysis of 1D fronts in series of images.

        Parameters
        ----------
        img_series : ImgSeries or ImgStack object
            image series on which the analysis will be run

        savepath : str or Path object
            folder in which to save analysis data & metadata
            (if not specified, the img_series savepath is used)
        """
        super().__init__(
            img_series=img_series,
            savepath=savepath,
        )

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
        front_data = img.mean(axis=0)
        data = {'analysis': front_data}
        return data
