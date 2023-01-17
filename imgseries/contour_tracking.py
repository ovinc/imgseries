"""Contour tracking on image series."""

# Standard library imports
import json

# Misc. package imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import measure
from tqdm import tqdm
import pandas as pd
from numpy import nan as NaN
import imgbasics

# Local imports
from .config import filenames
from .general import ImgSeries, Analysis
from .image_parameters import Contours


class ContourTracking(ImgSeries, Analysis):
    """Class to track contours on image series."""

    name = 'Images Series (ContourTracking)'

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
        self.contours = Contours(self)  # empty object

    def _find_contours(self, img, level):
        """Define how contours are found on an image."""
        image = img if img.ndim == 2 else self.rgb_to_grey(img)
        return measure.find_contours(image, level)

    # Basic analysis methods -------------------------------------------------

    def _contour_tracking(self, num, live):
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

        return data

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

    def _store_data(self, data):
        """Store contour and analysis data into dict/table."""

        num = data['num']
        n = len(data['analysis'])

        # Save contour data into dict ----------------------------------------
        for k in range(n):
            x, y = data['contours'][k]
            self.contour_data[k + 1][num] = {'x': list(x), 'y': list(y)}

        # Save analysis data into table --------------------------------------
        line = sum(data['analysis'], start=())  # "Flatten" list of tuples
        self.analysis_data.loc[num] = line

    def _run(self, num, live, blit=False):
        data = self._contour_tracking(num, live)
        self._update_reference_positions(data)
        self._store_data(data)
        if live:
            self._plot(data)
            if blit:
                return self.contour_lines + self.centroid_pts + [self.im]

    # Basic plot methods -----------------------------------------------------

    def _plot(self, data):

        img = data['image']
        num = data['num']

        self.ax.set_title(f'img #{num}, grey level {self.level}')

        if not self.plot_init_done:

            self.im = self.ax.imshow(img, cmap='gray')
            self.ax.axis('off')

            self.contour_lines = []
            self.centroid_pts = []

            for contour, analysis in zip(data['contours'], data['analysis']):

                contour_line, = self.ax.plot(*contour, '-r')
                self.contour_lines.append(contour_line)

                centroid_pt, = self.ax.plot(*analysis[:2], '+b')
                self.centroid_pts.append(centroid_pt)

            self.fig.tight_layout()
            self.plot_init_done = True

        else:

            self.im.set_array(img)

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

    # Public methods ---------------------------------------------------------

    def run(self, start=0, end=None, skip=1, live=False, blit=False):
        """Follow contours in an image series.

        Parameters
        ----------
        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - live: if True, plot detected contours on images in real time.

        Output
        ------
        Pandas dataframe with image numbers as the index, and the positions,
        perimeter and areas of the contours as a function of time as columns.
        """
        if self.contours.is_empty:
            msg = "Contours not defined yet. Use self.contours.define(), "\
                  "or self.contours.load() if contours have been previously saved."
            raise AttributeError(msg)

        self.level = self.contours.data['level']

        # Analysis parameters that will be saved into metadata file
        self.metadata['contours'] = self.contours.data

        nums = self.set_analysis_numbers(start, end, skip)

        self.reference_positions = list(self.contours.data['position'].values())
        n = len(self.reference_positions)

        # Initiate dict to store all contour data (for json saving later) ----

        self.contour_data = {k + 1: {} for k in range(n)}

        # Initiate pandas table to store data (for tsv saving later) ---------

        names = 'x', 'y', 'p', 'a'  # measurement names (p, a perimeter, area)
        cols = [name + str(k + 1) for k in range(n) for name in names]

        self.analysis_data = pd.DataFrame(index=nums, columns=cols)
        self.analysis_data.index.name = 'num'

        # Loop ---------------------------------------------------------------

        if not live:
            for num in tqdm(nums):
                self._run(num, live)
        else:
            self.plot_init_done = False
            self.fig, self.ax = plt.subplots()
            self.animation = FuncAnimation(fig=self.fig,
                                           func=self._run,
                                           fargs=(live, blit),
                                           frames=nums,
                                           cache_frame_data=False,
                                           repeat=False,
                                           blit=blit)

        # Finalization -------------------------------------------------------

        self.format_data(self.analysis_data)

    def save(self, filename=None):

        super().save(filename=filename)

        name = filenames[self.measurement_type] if filename is None else filename
        data_filename = name + '_Data'
        self._to_json(self.contour_data, data_filename)
