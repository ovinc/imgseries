"""Contour tracking on image series."""

# Misc. package imports
from skimage import measure
import pandas as pd
from numpy import nan as NaN
import matplotlib.pyplot as plt
import imgbasics

# Local imports
from .general import ImgSeries
from .analysis import Analysis
from .image_parameters import Contours
from .plot import ImagePlot


# ======================= Plotting / Animation classes =======================


class ContourTrackingPlot(ImagePlot):

    def create_plot(self):
        self.fig, self.ax = plt.subplots()

    def first_plot(self, data):
        """What to do the first time data arrives on the plot.

        self.updated_artists must be defined here.
        """
        img = data['image']
        num = data['num']

        self.ax.set_title(f'img #{num}, grey level {self.img_series.level}')
        self.imshow = self.ax.imshow(img, **self.kwargs)

        self.ax.axis('off')
        self.fig.tight_layout()

        self.contour_lines = []
        self.centroid_pts = []

        for contour, analysis in zip(data['contours'], data['analysis']):

            contour_line, = self.ax.plot(*contour, '-r')
            self.contour_lines.append(contour_line)

            centroid_pt, = self.ax.plot(*analysis[:2], '+b')
            self.centroid_pts.append(centroid_pt)

        self.updated_artists = self.contour_lines + self.centroid_pts + [self.imshow]

    def update_plot(self, data):
        """What to do upon iterations of the plot after the first time."""
        img = data['image']
        num = data['num']

        self.ax.set_title(f'img #{num}, grey level {self.img_series.level}')

        self.imshow.set_array(img)

        for contour, analysis, line, pt in zip(data['contours'],
                                               data['analysis'],
                                               self.contour_lines,
                                               self.centroid_pts):

            if contour is not None:
                line.set_data(*contour)
                pt.set_data(*analysis[:2])
            else:
                line.set_data(None, None)
                pt.set_data(None, None)


class ContourTrackingLivePlot(ContourTrackingPlot):

    def get_data(self, num):
        return self.img_series.live_analysis(num)


class ContourTrackingResultsPlot(ContourTrackingPlot):

    def get_data(self, num):
        return self.img_series.regenerate_data(num)


# =========================== Main ANALYSIS class ============================


class ContourTracking(ImgSeries, Analysis):
    """Class to track contours on image series."""

    name = 'Images Series (ContourTracking)'

    # Type of graph used for live view of analysis
    LivePlot = ContourTrackingLivePlot

    # Type of graph used for a-posteriori visualization of analysis
    # with animate() and inspect()
    Plot = ContourTrackingResultsPlot

    def __init__(self, paths='.', extension='.png', savepath='.', stack=None):
        """Init Contour Tracking analysis object.

        PARAMETERS
        ----------
        - paths: str, path object, or iterable of str/paths if data is stored
          in multiple folders.

        - extension: extension of image files (e.g. '.png')

        - savepath: path in which to save analysis files.

        If file series is in a stack rather than in a series of images:
        - stack: path to the stack (.tiff) file
          (parameters paths & extension will be ignored)
        """
        ImgSeries.__init__(self,
                           paths=paths,
                           savepath=savepath,
                           extension=extension,
                           stack=stack)

        Analysis.__init__(self, measurement_type='ctrack')

        # empty contour param object, needs to be filled with contours.define()
        # or contours.load() prior to starting analysis with self.run()
        self.contours = Contours(self)

    def _find_contours(self, img, level):
        """Define how contours are found on an image."""
        image = img if img.ndim == 2 else self.rgb_to_grey(img)
        return measure.find_contours(image, level)

    def _update_reference_positions(self, data):
        """Next iteration will look for contours close to the current ones."""
        for i, contour_analysis in enumerate(data['analysis']):
            if any(qty is NaN for qty in contour_analysis):
                # There has been a problem in detecting the contour
                pass
            else:
                # if position correctly detected, update where to look next
                xc, yc, *_ = contour_analysis
                self.reference_positions[i] = (xc, yc)

    def analyze(self, num, live=False):
        """Find contours at level in file i closest to the reference positions.

        Parameters
        ----------
        - num: file number identifier across the image file series
        - live: if True, plots detected contours on image

        Output
        ------
        [(x1, y1, p1, a1), (y2, y2, p2, a1), ..., (xn, yn, pn, an)] where n is the
        number of contours followed and (x, y), p, a is position, perimeter, area
        """
        img = self.read(num)
        contours = self._find_contours(img, self.level)

        data = {'analysis': []}     # Stores analysis data (centroid etc.)
        data['contours'] = []       # Stores full (x, y) contour data
        data['num'] = num

        if live:
            data['image'] = img

        for refpos in self.reference_positions:

            try:
                # this time edge=false, because trying to find contour closest
                # to the recorded centroid position, not edges
                contour = imgbasics.closest_contour(contours=contours,
                                                    position=refpos,
                                                    edge=True)

            except imgbasics.ContourError:
                # No contour at all detected on image --> return NaN
                xc, yc, perimeter, area = (NaN,) * 4
                contour = None

            else:

                x, y = imgbasics.contour_coords(contour, source='scikit')

                contprops = imgbasics.contour_properties(x, y)

                xc, yc = contprops['centroid']
                perimeter = contprops['perimeter']
                area = contprops['area']

            data['analysis'].append((xc, yc, perimeter, area))
            data['contours'].append((x, y))

        self._update_reference_positions(data)

        return data

    def initialize(self):
        """Check everything OK before starting analysis & initialize params."""

        if self.contours.is_empty:
            msg = "Contours not defined yet. Use self.contours.define(), "\
                  "or self.contours.load() if contours have been previously saved."
            raise AttributeError(msg)

        self.level = self.contours.data['level']

    def add_metadata(self):
        """Add useful analysis parameters etc. to the self.metadata dict.

        (later saved in the metadata json file)
        Define in subclasses
        """
        self.metadata['contours'] = self.contours.data

    def prepare_data_storage(self):
        """Prepare structure(s) that will hold the analyzed data."""
        self.reference_positions = list(self.contours.data['position'].values())
        n = len(self.reference_positions)

        # Initiate dict to store all contour data (for json saving later) ----

        self.contour_data = {str(k + 1): {} for k in range(n)}

        # Initiate pandas table to store data (for tsv saving later) ---------

        names = 'x', 'y', 'p', 'a'  # measurement names (p, a perimeter, area)
        cols = [name + str(k + 1) for k in range(n) for name in names]

        self.analysis_data = pd.DataFrame(index=self.nums, columns=cols)
        self.analysis_data.index.name = 'num'

    def store_data(self, data):
        """How to store data generated by analysis on a single image."""

        num = data['num']
        n = len(data['analysis'])

        # Save contour data into dict ----------------------------------------
        for k in range(n):
            x, y = data['contours'][k]
            # The str is because JSON converts to str, and so this makes
            # live data compatible with reloaded data from JSON
            self.contour_data[str(k + 1)][str(num)] = {'x': list(x), 'y': list(y)}

        # Save analysis data into table --------------------------------------
        line = sum(data['analysis'], start=())  # "Flatten" list of tuples
        self.analysis_data.loc[num] = line

    def generate_pandas_data(self):
        """How to convert data generated by store_data() into a pandas table."""
        return self.analysis_data

    def regenerate_data(self, num):
        """How to go back to raw dict of data from self.data.

        Useful for plotting / animating results again after analysis, among
        other things.
        """
        data = {'num': num}
        data['image'] = self.read(num=num)
        data['analysis'] = []
        data['contours'] = []

        n = len(self.contours.data['position'])

        for k in range(n):

            # contour positions and perimeters
            lim1 = 'x' + str(k + 1)
            lim2 = 'a' + str(k + 1)
            xc, yc, perimeter, area = self.data.loc[num, lim1:lim2]
            data['analysis'].append((xc, yc, perimeter, area))

            # full contour data
            x = self.contour_data[str(k + 1)][str(num)]['x']
            y = self.contour_data[str(k + 1)][str(num)]['y']
            data['contours'].append((x, y))

        return data

    def save(self, filename=None):
        """Save data and metadata into tsv/json files."""

        super().save(filename=filename)

        name = self._set_filename(filename)
        data_filename = name + '_Data'
        self._to_json(self.contour_data, data_filename)

    def regenerate(self, filename=None):
        """Save data and metadata into tsv/json files."""

        # Load data
        super().regenerate(filename=filename)

        # Load complete contour data
        name = self._set_filename(filename)
        data_filename = name + '_RawContourData'
        self.contour_data = self._from_json(data_filename)

        # regenerate internal contours object
        self.contours.load(filename=filename)

        # regenerate level at which contours are plotted
        self.level = self.contours.data['level']
