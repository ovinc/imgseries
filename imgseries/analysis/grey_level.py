"""Reflectance analysis: average grey level over time in img sequence."""

# Non-standard modules
import numpy as np
import matplotlib.pyplot as plt
import imgbasics

# Local imports
from ..parameters.analysis import Zones
from ..viewers import AnalysisViewer
from .formatters import PandasFormatter
from .results import PandasTsvJsonResults
from .analysis_base import Analysis


# ============================ Results formatting ============================


class GreyLevelFormatter(PandasFormatter):

    def _column_names(self):
        """Prepare structure(s) that will hold the analyzed data."""
        return self.analysis.zones.data

    def _data_to_results_row(self, data):
        """Generate iterable of data that fits in the defined columns."""
        return data['glevels']

    def _results_row_to_data(self, row):
        """Go from row of data to raw data"""
        return {'glevels': list(row)}

    def _to_metadata(self):
        """Get analysis metadata excluding paths and transforms"""
        return {
            'zones': self.analysis.zones.data,
            'function': f'{self.analysis.func.__module__}.{self.analysis.func.__name__}'
        }


class GreyLevelResults(PandasTsvJsonResults):
    default_filename = 'Img_GreyLevel'


# ======================== Plotting / Animation class ========================


class GreyLevelViewer(AnalysisViewer):

    # ---------------- Methods subclassed from AnalysisViewer ----------------

    def _create_figure(self):
        self.fig = plt.figure(figsize=(5, 7))
        xmin = 0.1
        xmax = 0.9
        w = xmax - xmin
        self.ax_img = self.fig.add_axes([xmin, 0.33, w, 0.65])
        self.ax_analysis = self.fig.add_axes([xmin, 0.09, w, 0.25])
        self.axs = self.ax_img, self.ax_analysis
        self.ax_img.axis('off')

    def _first_plot(self, data):
        """What to do the first time data arrives on the plot."""
        # Can be None when regenerating data and analysis not made for the
        # specific num
        glevels = data.get('glevels', None)

        self._create_image(data)

        self.current_pts = []
        self.live_pts = []
        self.bar = self.ax_analysis.axvline(x=data['num'], c='gray', alpha=0.5)
        self.zone_names = list(self.analysis.zones.data)

        for i, zone_name in enumerate(self.zone_names):

            color = self._get_color()

            # If existing results from previous analysis, show them as lines
            # The lines are static and are not modified later
            if self.analysis.results.data is not None:
                self.ax_analysis.plot(
                    self.analysis.results.data.index,
                    self.analysis.results.data[zone_name],
                    c=color,
                    alpha=0.5,
                )

            # If live, new analysis will be added as persisting dots
            # (not plotted on  first plot, but plotted from formatter data
            # during the animation / inspection)
            if self.live:
                live_pts, = self.ax_analysis.plot([], [], '.', c=color, alpha=0.5)
                self.live_pts.append(live_pts)

            # In any case, manage the incoming data, keeping in mind that
            # if data is re-generated from existing results, it might not be
            # available for the specific num and glevels can be None
            if glevels is not None:
                num = data['num']
                glevel = glevels[i]
            else:
                num = []
                glevel = []

            pt, = self.ax_analysis.plot(num, glevel, 'o', c=color, label=zone_name)
            self.current_pts.append(pt)

            # Plot cropzone on image
            zone = self.analysis.zones.data[zone_name]
            imgbasics.cropping._cropzone_draw(self.ax_img, zone, c=color)

        self.ax_analysis.legend()
        self.ax_analysis.grid()

        all_pts = self.current_pts + self.live_pts
        self.updated_artists = all_pts + [self.imshow, self.bar]

    def _update_plot(self, data):
        """What to do upon iterations of the plot after the first time."""
        num = data['num']
        glevels = data.get('glevels', None)

        self._update_image(data)
        self.bar.set_xdata(num)

        if glevels is None:
            return

        for zname, pt, glevel in zip(self.zone_names, self.current_pts, glevels):
            pt.set_data((num, glevel))

        if self.live:
            analysis_data = self.analysis.formatter.data
            for zname, live_pts in zip(self.zone_names, self.live_pts):
                live_pts.set_data(analysis_data.index, analysis_data[zname])

        self._autoscale(ax=self.ax_analysis)


# =========================== Main ANALYSIS class ============================


class GreyLevel(Analysis):
    """Class to perform analysis of average grey level on image series.

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
    Viewer = GreyLevelViewer
    Formatter = GreyLevelFormatter
    Results = GreyLevelResults

    # If results are independent (results from one num do not depend from
    # analysis on other nums), one do not need to re-do the analysis when
    # asking for the same num twice, and parallel computing is possible
    independent_results = True

    def __init__(
        self,
        img_series,
        savepath=None,
        func=np.mean,
    ):
        """Analysis of avg gray level on selected zone in series of images.

        img_series : ImgSeries or ImgStack object
            image series on which the analysis will be run

        savepath : str or Path object
            folder in which to save analysis data & metadata
            (if not specified, the img_series savepath is used)

        func : function
            function to be applied on the image pixels in the defined
            analysis zones (default: np.mean). Other typical functions
            can be: np.sum, np.max, etc.
        """
        super().__init__(img_series=img_series, savepath=savepath)

        # empty zones object, if not filled with zones.define() or
        # zones.load() prior to starting analysis with self.run(),
        # the whole image is considered
        self.zones = Zones(self)
        self.func = func

    # ------------------- Subclassed methods from Analysis -------------------

    def _init_analysis(self):
        """Check everything OK before starting analysis & initialize params."""
        if self.zones.is_empty:
            self._set_default_zone()

    def _analyze(self, img):
        """Analysis process on single image. Must return a dict.

        Parameters
        ----------
        img : array_like
            image array to be analyzed (e.g. numpy array).

        details : bool
            whether to include more details (e.g. for debugging or live view)


        Returns
        -------
        dict
            dict of data, handled by formatter._store_data()
        """
        glevels = []

        for cropzone in self.zones.data.values():
            img_crop = imgbasics.imcrop(img, cropzone)
            glevel = self.func(img_crop)
            glevels.append(glevel)

        data = {'glevels': glevels}

        return data

    # ---------------------------- Other methods -----------------------------

    def _set_default_zone(self):
        print('Warning: no zones defined; taking full image as default.')
        default_crop = 0, 0, self.img_series.nx, self.img_series.ny
        self.zones.data = {'zone 1': default_crop}

    # ------------------ Redefinitions of Analysis methods -------------------

    def regenerate(self, filename=None):
        """Load saved data, metadata and regenerate objects from them.

        Is used to reset the system in a state similar to the end of the
        analysis that was made before saving the results.

        Parameters
        ----------
        filename : str
            name of the analysis results file (if None, use default)

        Notes
        -----
            More or less equivalent to:
            >>> analysis.results.load(filename=filename)
            >>> image_series.load_transforms()
            (except that transforms are loaded from the metadata file of the
            analysis, not from a file generated by
            image_series.save_transforms())
        """

        # Load data
        super().regenerate(filename=filename)

        # regenerate internal zones object
        self.zones.load(filename=filename)
