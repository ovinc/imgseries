"""Analysis of image series (base class)"""

# Standard library imports
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Nonstandard
from tqdm import tqdm

# local imports
from .formatters import Formatter
from .results import Results
from ..config import CONFIG
from ..viewers import AnalysisViewer


class Analysis:
    """Base class for analysis subclasses (GreyLevel, ContourTracking, etc.)."""

    measurement_type = None  # define in subclasses (e.g. 'glevel', 'ctrack', etc.)

    DefaultViewer = AnalysisViewer     # redefine in subclasses
    DefaultFormatter = Formatter       # redefine in subclasses
    DefaultResults = Results           # redefine in subclasses

    def __init__(
        self,
        img_series,
        savepath=None,
        Viewer=None,
        Formatter=None,
        Results=None,
    ):
        """Initialize Analysis object

        Parameters
        ----------

        - img_series: image series from the ImgSeries class or subclasses

        - savepath: folder in which to save analysis data & metadata
                    (if not specified, the img_series savepath is used)

        - Viewer: Viewer class/subclasses that is used to display and inspect
                  analysis data (is used by ViewerTools)

        - Formatter: class/subclass of Formatter to format results spit out
                     by the raw analysis into something storable/saveable
                     by the Results class.

        - Results: Results class/subclasses that is used to store, save and
                   load analysis data and metadata.
        """
        Viewer = self.DefaultViewer if Viewer is None else Viewer
        Formatter = self.DefaultFormatter if Formatter is None else Formatter
        Results = self.DefaultResults if Results is None else Results

        self.img_series = img_series
        self.Viewer = Viewer
        self.formatter = Formatter(self)

        savepath = Path(savepath) if savepath else img_series.savepath
        self.results = Results(savepath=savepath)

        if self.img_series.is_stack:
            # Below, pre-populate parameters to be saved as metadata.
            # Other metadata will be added to this dict before saving
            # into metadata file
            stack_path = os.path.relpath(self.img_series.path, savepath)
            self.results.metadata['stack'] = stack_path
        else:
            folders = [os.path.relpath(f, savepath) for f in self.img_series.folders]
            self.results.metadata['path'] = str(savepath.resolve()),
            self.results.metadata['folders'] = folders

    # ========================= Misc. internal tools =========================

    def _add_transform_to_metadata(self):
        """Add information about image transforms (rotation, crop etc.) to metadata."""
        for transform_name in CONFIG['image transforms']:
            transform_data = getattr(self.img_series, transform_name).data
            self.results.metadata[transform_name] = transform_data

    def _analyze_live(self, num):
        data = self.analyze(num=num, live=True)
        self.formatter._store_data(data)
        return data

    # ============================ Public methods ============================

    def run(
        self,
        start=0,
        end=None,
        skip=1,
        parallel=False,
        nprocess=None,
        live=False,
        blit=False
    ):
        """Start analysis of image sequence.

        PARAMETERS
        ----------
        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - parallel: if True, distribute computation across different processes.
          (only available if calculations on each image is independent of
          calculations on the other images)

        - nprocess: number of process workers; if None (default), use default
          in ProcessPoolExecutor, depends on the number of cores of computer)

        - live: if True, plot analysis results in real time.
        - blit: if True, use blitting to speed up live display

        OUTPUT
        ------
        Pandas dataframe with image numbers as the index, and with columns
        containing timestamps and the calculated data.

        WARNING
        -------
        If running on a Windows machine and using the parallel option, the
        function call must not be run during import of the file containing
        the script (i.e. the function must be in a `if __name__ == '__main__'`
        block). This is because apparently multiprocessing imports the main
        program initially, which causes recursive problems.
        """
        self.nums = self.img_series._set_substack(start, end, skip)
        self.nimg = len(self.nums)

        self._initialize()
        self._add_metadata()
        self._add_transform_to_metadata()

        self.formatter._prepare_data_storage()

        if parallel:  # ================================= Multiprocessing mode

            futures = {}

            with ProcessPoolExecutor(max_workers=nprocess) as executor:

                for num in self.nums:
                    future = executor.submit(self.analyze, num, live=False)
                    futures[num] = future

                # Waitbar ----------------------------------------------------
                futures_list = list(futures.values())
                for future in tqdm(as_completed(futures_list), total=self.nimg):
                    pass

                # Get results ------------------------------------------------
                for num, future in futures.items():
                    data = future.result()
                    self.formatter._store_data(data)

        else:  # ============================================= Sequential mode

            if not live:
                for num in tqdm(self.nums):
                    data = self.analyze(num=num, live=False)
                    self.formatter._store_data(data)
            else:
                # plot uses self._analyze_live to calculate and store data
                live_plot = self.Viewer(self, live=True)
                # without self.animation, the animation is garbage collected
                self.animation = live_plot.animate(nums=self.nums, blit=blit)

        # Finalize and format data -------------------------------------------

        # if live, it's the _on_fig_close() method of the viewer which takes
        # care of saving the data, because if not, save_results() is called
        # at the beginning of the FuncAnimation (i.e., analysis in this case,
        # and no data is saved)
        if not live:
            self.formatter._to_results()

    def regenerate(self, filename=None):
        """Load saved data, metadata and regenerate objects from them.

        Is used to reset the system in a state similar to the end of the
        analysis that was made before saving the results.

        Parameters
        ----------
        - filename: name of the analysis results file (if None, use default)

        More or less equivalent to:
        analysis.results.load(filename=filename)
        image_series.load_transforms()
        (except that transforms are loaded from the metadata file of the
        analysis, not from a file generated by image_series.save_transforms())
        """
        # load data from files
        self.results.load(filename=filename)

        # re-apply transforms (rotation, crop etc.)

        for transform_name in CONFIG['image transforms']:

            # e.g. self.img_series.crop.reset()
            getattr(self.img_series, transform_name).reset()

            # e.g. self.img_series.crop.data = self.results.metadata['crop']
            data = self.results.metadata.get(transform_name, {})
            setattr(getattr(self.img_series, transform_name), 'data', data)

    # ==================== Interactive inspection methods ====================

    # Note: Initially, these were in a ViewerTools subclass to avoid code
    # repetition, but I eventually preferred to repeat code to avoid
    # multiple inheritance and weird couplings.

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series

        - transform: if True (default), apply active transforms
                     if False, load raw image.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.show(num=num)

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image series.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self.img_series._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=nums)

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - blit: if True, use blitting for faster animation.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        nums = self.img_series._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.animate(nums=nums, blit=blit)

    # =================== Methods to define in subclasses ====================

    def _initialize(self):
        """Check everything OK before starting analysis & initialize params.

        Define in subclasses."""
        pass

    def _add_metadata(self):
        """Add useful analysis parameters etc. to the self.metadata dict.

        (later saved in the metadata json file)
        Define in subclasses."""
        pass

    def _analyze(self, img):
        """Analysis process on single image. Must return a dict.

        Parameters
        ----------
        - img: image array to be analyzed (e.g. numpy array).

        Output
        ------
        - dict of data, handled by formatter._store_data()

        Define in subclasses."""
        pass

    def analyze(self, num, live=False):
        """Same as _analyze, but with num as input instead of img.

        Can be subclassed if necessary.

        Parameters
        ----------
        - num: file number identifier across the image file series
        - live: if True, add image to data for live visualization

        Output
        ------
        - dict of data, handled by formatter._store_data()

        Define in subclasses."""
        img = self.img_series.read(num=num)
        data = self._analyze(img=img)
        data['num'] = num
        if live:
            data['image'] = img
        return data
