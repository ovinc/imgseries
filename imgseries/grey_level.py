"""Reflectance analysis: average grey level over time in img sequence."""

# Standard Library
from concurrent.futures import ProcessPoolExecutor, as_completed

# Non-standard modules
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .config import _crop
from .general import ImgSeries, Analysis
from .image_parameters import Zones


class GreyLevel(ImgSeries, Analysis):
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
        ImgSeries.__init__(self,
                           paths=paths,
                           savepath=savepath,
                           extension=extension,
                           stack=stack)

        Analysis.__init__(self, measurement_type='glevel')

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
            img_crop = _crop(img, cropzone)
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
        self.metadata['zones'] = self.zones.data

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
