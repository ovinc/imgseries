# Misc. package imports
#from matplotlib.lines import _LineStyle
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import pandas as pd
from numpy import nan as NaN
import imgbasics

from scipy import ndimage
import imageio as iio
from scipy import optimize as opt
import matplotlib.pyplot as plt
import numpy as np
import glob as gl
from scipy.signal import savgol_filter
import matplotlib.patches as patches

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.filters import gaussian


# Local imports
from .general import ImgSeries

def _azimuthal_average(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def _find_circular_contour(img, *args):
    """Method to find the imbibition contour:
    Hough_circular, is the method to detect a circular contour
    """
    sigma, level_down, level_high = args[0], args[1], args[2]
    cany, hough_radii = args[3], args[4]

    # Filters
    # Remove noise from the image with a gausian filter --------------
    img_blur = gaussian(img, sigma=sigma)

    # Homemade theshold
    img_th = 255 * np.ones(img_blur.shape)  # 2**8 -1 bits
    img_th[(level_down < img_blur) & (img_blur < level_high)] = 1
    img_th[(img_blur < level_down)] = 0
    img_th[(img_blur > level_high)] = 0

    # Canny
    edges = canny(img_th, **cany)

    # Circular imbibition contour
    # Hough transformation to determine imbibition front contour
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    _, cx, cy, radius = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=1)

    circle = (radius, cx[0], cy[0])

    return img_blur, edges, hough_res[0], circle

# ==================== Class to manage reference contours ====================

class ImbibitionFront:
    """Class to store and manage reference contours param in image series."""

    def __init__(self, img_series):
        """Init Contours object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. ContourTracking)
        """
        self.img_series = img_series
        self.img_ref = None
        self.data_method = {}
        self.data_image = {}
        self.reference = None # True if n_start, n_indexes exist

    def _references_defaults(self, crop):
        """ Determine the reference image (an average of the references)
        as well as te number at which the experiment begins
        """

        n_files = len(self.img_series.files)
        # the experiment starts
        im_mean = []
        im_std_crop = []
        for i in range(n_files):
            img = self.img_series.read(i)
            im_mean.append(img.mean())

            img_crop = imgbasics.imcrop(img, crop)
            im_std_crop.append(img_crop.std())

        # Determination of n_start per default
        im_mean = np.array(im_mean)
        mean = im_mean.mean()
        bol = (im_mean < mean).astype(int)
        n_start = np.where(np.diff(bol) == 1)[0][0] + 1

        # Determination of the end images per default
        im_std_crop = np.array(im_std_crop)
        n_end = 10  # select the last 10 images
        mean_crop = im_std_crop[-n_end:].mean()
        # Around 10% of the mean value (arbitrary choice)
        condition = (im_std_crop < mean_crop * 1.1) & (0.99 * mean_crop < im_std_crop)
        n_indexes = np.where(condition == True)[0]

        img_refs = []
        for num in n_indexes:
            img_refs.append(self.img_series.read(num))

        self.img_ref = np.stack(img_refs, axis=2).mean(axis=2)
        self.reference = n_start, n_indexes

    def set_reference(self, n_refs=None, n_start=None):
        """
        Calculate the reference image

        Example:
        n_refs = 100  # number of images over which to average at end of movie
        images = range(-n_refs, 0) start from n to the last image
        """
        img_refs = []
        n_indexes = range(-n_refs, 0)
        for i in n_indexes:
            img_refs.append(self.img_series.read(i))

        self.img_ref = np.stack(img_refs, axis=2).mean(axis=2)
        self.reference = n_start, n_indexes

    def define(self, num=0,
               sigma=None, level_down=None, level_high=None,
               hough_radii=None, cany=None):
        """Interactively define n contours on an image at level level.

        Parameters
        ----------
        - images: range of image where a an average is performed
        example(images = range(-n, 0) start from -n to the last image)
        - sigma: sigma used for the gaussian filter
        - level_down level_up: levels to define threshold to detect contours
        between 0 and 1 since gaussian filter is performed
        - n: number of contours
        - num: image identifier (num=0 corresponds to first image in first folder)

        Output
        ------
        None, but stores in self.data a dictionary with keys:
        'crop', 'position', 'level', 'image'
        """
        # Image
        img = self.img_series.read(num=num)

        # Crop image
        _, crop = imgbasics.imcrop(img)

        if self.reference is None:
            self._references_defaults(crop)

        img_ref = self.img_ref
        n_start, n_indexes = self.reference

        # substraction to get a better resolution of the imbibition front
        img = (img - img_ref) / img_ref

        # Filters: see ipython module to understand these default settingd
        # Remove noise from the image with a gausian filter --------------
        if sigma is None:
            sigma = 3

        # Homemade theshold
        if level_down is None:
            level_down = -0.10117
        if level_high is None:
            level_high = -0.095  #-0.1

        # Circular imbibition contour
        # Hough transformation to determine imbibition front contour
        if hough_radii is None:
            hough_radii = np.arange(250, 350, 10)

        if cany is None:
            cany = {'sigma': 3, 'low_threshold': 0, 'high_threshold': 0.95}

        # egdes from the image with
        self.data_image = {'crop': crop,
                           'image': num,
                           'number start': int(n_start),
                           'img_ref': img_ref,
                           'ref': [f'{value}' for value in n_indexes]}

        self.data_method = {'level_down': level_down,
                            'level_high': level_high,
                            'sigma': sigma,
                            'hough_radii start': [f'{value}' for value in hough_radii],
                            'canny': cany
                            }

    def show(self, **kwargs):
        """Show reference contours used for contour tracking.
        with the circular ough_transformation
        Parameters
        ----------
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default)
        """
        # Image
        num = self.data_image['image']
        crop = self.data_image['crop']
        img_ref = self.img_ref

        # Method
        level_down = self.data_method['level_down']
        level_high = self.data_method['level_high']
        sigma = self.data_method['sigma']
        hough_radii = np.array(self.data_method['hough_radii start']).astype(int)
        cany = self.data_method['canny']

        # Load image, crop it, and calculate contours
        img = (self.img_series.read(num) - img_ref) / img_ref
        img_crop = imgbasics.imcrop(img, crop)

        _, edges, hough, circle = _find_circular_contour(img_crop,
                                                         sigma, level_down,
                                                         level_high, cany,
                                                         hough_radii)


        theta = np.arange(0, 360, 0.5)
        r, x0, y0 = circle[0], circle[1], circle[2]
        x = x0 + r * np.cos(theta)
        y = y0 + r * np.sin(theta)

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1,
                                            constrained_layout=True,
                                            figsize=(17, 8))

        ax1.set_title('image and resulting circular contour')
        ax1.imshow(img_crop, **kwargs)
        ax1.plot(x, y, '.', markersize=1, linewidth=1)
        ax1.plot(x0, y0, 'r+')

        ax2.set_title('edges resulting from successive filters\n\
        (gaussian, homemade threshold and canny)')
        ax2.imshow(edges)

        ax3.set_title('hough transformation space')
        ax3.imshow(hough)

        fig.suptitle(f'img #{num}')

        plt.show()

        return ax1

    def load(self, filename=None):
        """Load contour data from .json file and put it in self.data."""
        self.data_method = self.img_series.load_metadata(filename=filename)['imbibition']['method']
        self.data_image = self.img_series.load_metadata(filename=filename)['imbibition']['image']
        n_indexes = np.array(self.data_image['ref']).astype(np.int64)
        img_refs = []
        for num in n_indexes:
            img_refs.append(self.img_series.read(num))

        self.img_ref = np.stack(img_refs, axis=2).mean(axis=2)
        self.data_image['img_ref'] = self.img_ref

    @property
    def is_empty(self):
        status = True if self.data_method == {} else False
        return status

# ==================== Class for Contour Tracking Analysis ===================


class ImbibitionTracking(ImgSeries):
    """Class to track Imbibition contours on image series."""

    name = 'Images Series (ImbibitionTracking)'

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
                         measurement_type='itrack', stack=stack)

        # empty contour param object, needs to be filled with contours.define()
        # or contours.load() prior to starting analysis with self.run()
        self.imbibition = ImbibitionFront(self)  # empty object

    # Basic analysis method --------------------------------------------------

    def _imbibition_tracking(self, num, live):
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
        img_ref = self.img_ref
        img = (self.read(num) - img_ref) / img_ref
        # Choose one channel arbitrary 1
        img_crop = imgbasics.imcrop(img, self.crop)

        # Method
        level_down = self.data_method['level_down']
        level_high = self.data_method['level_high']
        sigma = self.data_method['sigma']
        if self.hough_radii is None:
            self.hough_radii = np.array(self.data_method['hough_radii start']).astype(np.int64)
        cany = self.data_method['canny']

        # method hough
        img_blur, _, _, circle = _find_circular_contour(img_crop,
                                                        sigma, level_down,
                                                        level_high, cany,
                                                        self.hough_radii)

        r, x0, y0 = circle[0][0], circle[1], circle[2]
        radii = np.arange(r - 20, r + 20, 1)
        self.hough_radii = radii[radii > 20]  # new estimation of range radii

        # method integration
        avg = _azimuthal_average(img_blur, center=(x0, y0))
        r_pixel = np.arange(0, len(avg))
        dr = np.diff(r_pixel).mean()
        condition_min = (np.max(np.abs(avg[abs(r - r_pixel) < dr * 40])) == np.abs(avg))  # arbitrary
        r_min = r_pixel[condition_min]
        avg_min = avg[condition_min]
        condition_r = (r_pixel > r_min - 40) & (r_pixel < r_min + 40)
        avg_r = avg[condition_r]
        r_r = r_pixel[condition_r]
        condition = np.abs(avg_r) > (np.abs(avg_r[r_min == r_r]) * 0.9)  # arbitrary choice
        r_width = r_r[condition]

        if live:
            theta = np.arange(0, 360, 0.5)
            x = x0 + r_min * np.cos(theta)
            y = y0 + r_min * np.sin(theta)
            self.ax1.clear()
            self.ax1.imshow(img_crop, vmin=-0.1, vmax=0.2)
            self.ax1.axis('off')
            self.ax1.set_title(f'img #{num}')
            self.ax1.plot(x, y, '.', markersize=1, linewidth=1)    # contour
            self.ax1.plot(x0, y0, 'r+')  # centroid position

            self.ax2.clear()
            self.ax2.plot(r_min, avg_min, 'kd')
            self.ax2.plot(r_pixel, avg, '.-')
            self.ax2.axvline(x=r_min)
            self.ax2.axvline(x=r_width[0],
                             color='k', linestyle='--')
            self.ax2.axvline(x=r_width[-1], color='k',
                             linestyle='--')
            self.ax2.set_xlabel('radius (index)')
            self.ax2.set_ylabel('intensity (au)')

            plt.pause(0.001)

        return (r_min[0], x0, y0, r_width[0], r_width[-1])

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
        if self.imbibition.is_empty:
            msg = "Imbibtion contour not defined yet. Use self.imbibition.define(), "\
                  "or self.imbibition.load() if contours have been previously saved."
            raise AttributeError(msg)

        self.data_method = self.imbibition.data_method
        self.data_image = self.imbibition.data_image
        self.crop = self.data_image['crop']
        self.img_ref = self.imbibition.img_ref

        # Analysis parameters that will be saved into metadata file
        self.parameters['imbibition'] = {}
        self.parameters['imbibition']['method'] = {x: self.data_method[x]
                                                   for x in self.data_method}
        self.parameters['imbibition']['image'] = {x: self.data_image[x]
                                                  for x in self.data_image
                                                  if x not in ['img_ref']}

        nums = self.set_analysis_numbers(start, end, skip)

        # Initiate pandas table to store data --------------------------------

        cols = ['radi', 'centerx', 'centery', 'r_min', 'r_max']

        data = pd.DataFrame(index=nums, columns=cols)
        data.index.name = 'num'

        if live:
            fig, (self.ax1, self.ax2) = plt.subplots(ncols=2, nrows=1,
                                                     constrained_layout=True)

        # Loop ---------------------------------------------------------------
        # Initialisation
        self.hough_radii = None
        for num in tqdm(nums):

            try:
                track = self._imbibition_tracking(num, live)
            except IndexError:
                print(f'no detection of the contour num={num}')

            else:
                data.loc[num] = track


        self.format_data(data)
