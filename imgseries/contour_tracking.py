"""Contour tracking on image series."""

# Misc. package imports
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import pandas as pd
from numpy import nan as NaN
import imgbasics

# Local imports
from .general import ImgSeries


# ==================== Class to manage reference contours ====================

class Contours:
    """Class to store and manage reference contours param in image series."""

    def __init__(self, img_series):
        """Init Contours object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. ContourTracking)
        """
        self.img_series = img_series
        self.data = {}

    def define(self, level, n=1, num=0):
        """Interactively define n contours on an image at level level.

        Parameters
        ----------
        - level: grey level at which to define threshold to detect contours
        - n: number of contours
        - num: image identifier (num=0 corresponds to first image in first folder)

        Output
        ------
        None, but stores in self.data a dictionary with keys:
        'crop', 'position', 'level', 'image'
        """
        img = self.img_series.read(num=num)

        img_crop, crop = imgbasics.imcrop(img)
        contours = measure.find_contours(img_crop, level)

        # Display the cropped image and plot all contours found --------------

        fig, ax = plt.subplots()
        ax.imshow(img_crop, cmap='gray')
        ax.set_xlabel('Left click on vicinity of contour to select.')

        for contour in contours:
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        # Interactively select contours of interest on image -----------------

        positions = {}

        for k in range(1, n + 1):

            ax.set_title(f'Contour {k} / {n}')
            fig.canvas.draw()
            fig.canvas.flush_events()

            pt, = plt.ginput()

            contour = imgbasics.closest_contour(contours, pt, edge=True)
            x, y = imgbasics.contour_coords(contour, source='scikit')

            ax.plot(x, y, linewidth=1, c='y')
            plt.pause(0.01)

            xc, yc = imgbasics.contour_properties(x, y)['centroid']

            name = f'contour {k}'
            positions[name] = (xc, yc)  # store position of centroid

        plt.close(fig)

        self.data = {'crop': crop,
                     'position': positions,
                     'level': level,
                     'image': num}

    def show(self, **kwargs):
        """Show reference contours used for contour tracking.

        Parameters
        ----------
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default)
        """
        num = self.data['image']
        level = self.data['level']
        crop = self.data['crop']
        positions = self.data['position']

        # Load image, crop it, and calculate contours
        img = self.img_series.read(num)
        img_crop = imgbasics.imcrop(img, crop)
        contours = measure.find_contours(img_crop, level)

        fig, ax = plt.subplots()

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'gray'
        ax.imshow(img_crop, **kwargs)

        # Find contours closest to reference positions and plot them
        for contour in contours:
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=1, c='b')

        # Interactively select contours of interest on image -----------------
        for pt in positions.values():
            contour = imgbasics.closest_contour(contours, pt, edge=False)
            x, y = imgbasics.contour_coords(contour, source='scikit')
            ax.plot(x, y, linewidth=2, c='r')

        ax.set_title(f'img #{num}, grey level {level}')

        plt.show()

        return ax

    def load(self, filename=None):
        """Load contour data from .json file and put it in self.data."""
        self.data = self.img_series.load_metadata(filename=filename)['contours']

    @property
    def is_empty(self):
        status = True if self.data == {} else False
        return status

# ==================== Class for Contour Tracking Analysis ===================


class ContourTracking(ImgSeries):
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
        super().__init__(paths=paths, savepath=savepath, extension=extension,
                         measurement_type='ctrack', stack=stack)

        # empty contour param object, needs to be filled with contours.define()
        # or contours.load() prior to starting analysis with self.run()
        self.contours = Contours(self)  # empty object

    # Basic analysis method --------------------------------------------------

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
        img_crop = imgbasics.imcrop(img, self.crop)

        if live:
            self.ax.clear()
            self.ax.imshow(img_crop, cmap='gray')
            self.ax.axis('off')
            self.ax.set_title(f'img #{num}, grey level {self.level}')

        contours = measure.find_contours(img_crop, self.level)

        data = []

        for refpos in self.reference_positions:

            try:
                # this time edge=false, because trying to find contour closest
                # to the recorded centroid position, not edges
                contour = imgbasics.closest_contour(contours, refpos, edge=True)

            except imgbasics.ContourError:
                # No contour at all detected on image --> return NaN
                xypa = (NaN, NaN, NaN, NaN)

            else:

                x, y = imgbasics.contour_coords(contour, source='scikit')

                contprops = imgbasics.contour_properties(x, y)

                xc, yc = contprops['centroid']
                perimeter = contprops['perimeter']
                area = contprops['area']

                if live:
                    self.ax.plot(x, y, '-r')    # contour
                    self.ax.plot(xc, yc, '+b')  # centroid position

            data.append((xc, yc, perimeter, area))

        if live:
            plt.pause(0.001)

        return data

    # Public methods --------------------------------------------------------

    def run(self, start=0, end=None, skip=1, live=False):
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
        self.crop = self.contours.data['crop']

        # Analysis parameters that will be saved into metadata file
        self.parameters['contours'] = self.contours.data

        nums = self.set_analysis_numbers(start, end, skip)

        self.reference_positions = list(self.contours.data['position'].values())
        n = len(self.reference_positions)

        # Initiate pandas table to store data --------------------------------

        names = 'x', 'y', 'p', 'a'  # measurement names (p, a perimeter, area)
        cols = [name + str(k + 1) for k in range(n) for name in names]

        data = pd.DataFrame(index=nums, columns=cols)
        data.index.name = 'num'

        if live:
            fig, self.ax = plt.subplots()

        # Loop ---------------------------------------------------------------

        for num in tqdm(nums):

            track = self._contour_tracking(num, live)

            # next iteration will look for contours close to the current ones

            for i, contour_data in enumerate(track):
                if any(qty is NaN for qty in data):
                    # There has been a problem in detecting the contour
                    pass
                else:
                    # if position correctly detected, update where to look next
                    xc, yc, *_ = contour_data
                    self.reference_positions[i] = (xc, yc)

            # Save data into table
            line = sum(track, start=())  # "Flatten" list of tuples
            data.loc[num] = line


        self.format_data(data)
