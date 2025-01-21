"""Class ImgSeries for image series manipulation"""

# Standard library imports
from functools import lru_cache

# Nonstandard
import filo

# local imports
from ..config import CONFIG
from ..fileio import FileIO
from ..viewers import ImgSeriesViewer

from .general import ImgSeriesBase, ImageReader


class ImgSeriesReader(ImageReader):

    @staticmethod
    def _read_image(file):
        return FileIO.read_single_image(file=file)

    def _read(self, num):
        """read raw image from image series"""
        file = self.img_series.files[num].file
        return self._read_image(file)


class ImgSeries(ImgSeriesBase, filo.Series):
    """Class to manage series of images, possibly in several folders."""

    # Only for __repr__ (str representation of class object, see filo.Series)
    name = 'Image Series'

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = CONFIG['filenames']['files'] + '.tsv'

    # To distinguish between ImgSeries and ImgStack
    is_stack = False

    def __init__(
        self,
        paths='.',
        extension='.png',
        savepath='.',
        corrections=None,
        transforms=None,
        correction_order=None,
        transform_order=None,
        Viewer=ImgSeriesViewer,
        ImgReader=ImgSeriesReader,
    ):
        """Init image series object.

        Parameters
        ----------
        paths : str, path object or iterable of those
            can be a string, path object, or a list of str/paths if data
            is stored in multiple folders.

        extension : str
            extension of files to consider (e.g. '.png')

        savepath: str or path object
            folder in which to save parameters (transform, display etc.)

        corrections : dict
            with keys: correction names and values: correction classes

        transforms : dict
            with keys: transform names and values: transform classes

        correction_order : iterable
            iterable of names of corrections to consider (their order indicates
            the order in which they are applied),
            e.g. corrections=('flicker', 'shaking')

        transform_order : iterable
            iterable of names of transforms to consider (their order indicates
            the order in which they are applied),
            e.g. transforms=('rotation', 'crop', 'filter')

        Viewer : class
            which Viewer class to use for show(), inspect() etc.

        ImgReader: class
            class (or object) that defines how to read images
        """
        # Inherit useful methods and attributes for file series
        # (including self.savepath)
        filo.Series.__init__(
            self,
            paths=paths,
            extension=extension,
            savepath=savepath,
        )

        ImgSeriesBase.__init__(
            self,
            corrections=corrections,
            transforms=transforms,
            correction_order=correction_order,
            transform_order=transform_order,
            Viewer=Viewer,
            ImgReader=ImgReader,
        )

        self._get_initial_image_dims()

    @property
    def nums(self):
        """Iterator (sliceable) of image identifiers.

        Allows the user to do e.g.
        ```python
        for num in images.nums[::3]:
            images.read(num)
        ```
        """
        return range(len(self.files))


# ----------------------------------------------------------------------------
# ============== Factory function to generate ImgSeries objects ==============
# ----------------------------------------------------------------------------


def series(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option."""
    if not cache:

        return ImgSeries(*args, **kwargs)

    else:

        class ImgSeriesCached(ImgSeries):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgSeriesCached(*args, **kwargs)
