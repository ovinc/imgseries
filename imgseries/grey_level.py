"""Reflectance analysis: average grey level over time in img sequence."""

# Standard Library
from concurrent.futures import ProcessPoolExecutor, as_completed

# Non-standard modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import imgbasics
from imgbasics.cropping import _cropzone_draw

# Local imports
from .general import ImgSeries


# ==================== Class to manage zones of interest =====================


class Zones:
    """Class to store and manage areas of interest on series of images."""

    def __init__(self, img_series):
        """Init Zones object.

        Parameters
        ----------
        - img_series: object of an image series class (e.g. GreyLevel)
        """
        self.img_series = img_series  # ImgSeries object on which to define zones
        self.data = {}  # dict {'zone 1": (x, y, w, h), 'zone 2': ... etc.}

    def define(self, n=1, num=0, draggable=False):
        """Interactively define n zones image.

        Parameters
        ----------
        - n: number of zones to analyze (default 1)

        - num: image ('num' id) on which to select crop zones. Note that
          this number can be different from the name written in the image
          filename, because num always starts at 0 in the first folder.

        - draggable: use draggable rectangle from drapo to define crop zones
          instead of clicking to define opposite rectangle corners.

        Output
        ------
        None, but stores in self.data a dict with every cropzone used during
        the analysis, with:
        Keys: 'zone 1', 'zone 2', etc.
        Values: tuples (x, y, width, height)
        """
        img = self.img_series.read(num=num)

        if img.ndim > 2:
            kwargs = {}
        else:
            kwargs = {'cmap': 'gray'}

        fig, ax = plt.subplots()
        ax.imshow(img, **kwargs)
        ax.set_title('All zones defined so far')

        zones = {}

        for k in range(1, n + 1):

            msg = f'Select zone {k} / {n}'

            _, cropzone = imgbasics.imcrop(img,
                                           message=msg,
                                           draggable=draggable,
                                           **kwargs)

            name = f'zone {k}'
            zones[name] = cropzone
            _cropzone_draw(ax, cropzone, c='b')

        plt.close(fig)

        self.data = zones

    def show(self, num=0, **kwargs):
        """show the defined zones on image (image id num if specified)

        Parameters
        ----------
        - num: id number of image on which to show the zones (default first one).
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default)
        """
        ax = self.img_series.show(num, **kwargs)
        ax.set_title(f'Analysis Zones (img #{num})')

        for zone in self.data.values():
            _cropzone_draw(ax, zone, c='r')

        return ax

    def load(self, filename=None):
        """Load zones data from .json file and put it in zones.data.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.data = self.img_series.load_metadata(filename=filename)['zones']

    @property
    def is_empty(self):
        status = True if self.data == {} else False
        return status


# ================================ Main Class ================================


class GreyLevel(ImgSeries):
    """Class to perform analysis of average grey level on image series."""

    name = 'Images Series (GreyLevel)'  # used for __repr__

    def __init__(self, paths='.', extension='.png', savepath='.', stack=None):
        """Analysis of avg gray level on selected zone in series of images.

        Parameters
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
                         measurement_type='glevel', stack=stack)

        # empty zones object, needs to be filled with zones.define() or
        # zones.load() prior to starting analysis with self.run()
        self.zones = Zones(self)

    # Basic analysis method --------------------------------------------------

    def _gray_analysis(self, num):
        """Basic analysis function, to be threaded or multiprocessed.

        Parameters
        ----------
        - num: file number identifier across the image file series

        Output
        ------
        """
        glevels = []
        img = self.read(num)
        for cropzone in self.zones.data.values():
            img_crop = imgbasics.imcrop(img, cropzone)
            glevel = np.mean(img_crop)
            glevels.append(glevel)
        return glevels

    # Public methods ---------------------------------------------------------

    def run(self, start=0, end=None, skip=1, parallel=False, nprocess=None):
        """Analysis of avg gray level on selected zone(s) in series of images.

        PARAMETERS
        ----------
        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - parallel: if True, distribute computation across different processes.

        - nprocess: number of process workers; if None (default), use default
          in ProcessPoolExecutor, depends on the number of cores of computer)

        OUTPUT
        ------
        Pandas dataframe with image numbers as the index, and the various zones
        analyzed as columns, with names such as 'zone 7' (numbering starts at 1)

        WARNING
        -------
        If running on a Windows machine and using the parallel option, the
        function call must not be run during import of the file containing
        the script (i.e. the function must be in a `if __name__ == '__main__'`
        block). This is because apparently multiprocessing imports the main
        program initially, which causes recursive problems.
        """
        if self.zones.is_empty:
            msg = "Analysis zones not defined yet. Use self.zones.define(),  "\
                  "or self.zones.load() if zones have been previously saved."
            raise AttributeError(msg)

        # Analysis parameters that will be saved into metadata file
        self.parameters['zones'] = self.zones.data

        nums = self.set_analysis_numbers(start, end, skip)
        nimg = len(nums)

        glevel_data = []

        if parallel:  # ================================= Multiprocessing mode

            futures = {}

            with ProcessPoolExecutor(max_workers=nprocess) as executor:

                for num in nums:
                    future = executor.submit(self._gray_analysis, num)
                    futures[num] = future

                # Waitbar ----------------------------------------------------
                futures_list = list(futures.values())
                for future in tqdm(as_completed(futures_list), total=nimg):
                    pass

                # Get results ------------------------------------------------
                for num, future in futures.items():
                    glevels = future.result()
                    glevel_data.append(glevels)

        else:  # ============================================= Sequential mode

            for num in tqdm(nums):
                glevels = self._gray_analysis(num)
                glevel_data.append(glevels)

        # Format data --------------------------------------------------------

        zone_names = self.zones.data.keys()  # 'zone 1', 'zone 2', etc.
        data = pd.DataFrame(glevel_data, index=nums, columns=zone_names)
        data.index.name = 'num'

        self.format_data(data)
