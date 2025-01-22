"""Class ImgSeries for image series manipulation"""

# Standard library imports
from functools import lru_cache

# Nonstandard
import filo

# local imports
from ..config import CONFIG
from ..readers import SingleImageReader
from ..viewers import ImgSeriesViewer

from .image_base import ImgSeriesBase


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
        ImgViewer=ImgSeriesViewer,
        ImgReader=SingleImageReader,
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

        corrections : iterable of str
            iterable of names of corrections to consider
            (their order indicates the order in which they are applied),
            e.g. corrections=('flicker', 'shaking');
            if None, use default order.

        transforms : iterable of str
            iterable of names of transforms to consider
            (their order indicates the order in which they are applied),
            e.g. transforms=('rotation', 'crop', 'filter');
            if None, use default order.

        ImgViewer : subclass of ImageViewerBase
            which Viewer class to use for show(), inspect() etc.
            if None, use default viewer class

        ImgReader : subclass of ImageReaderBase
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
            ImgViewer=ImgViewer,
            ImgReader=ImgReader,
        )

        self._get_initial_image_dims()

    @property
    def nums(self):
        """Iterator (sliceable) of image identifiers.

        Examples
        --------
        Allows the user to do e.g.
        >>> for num in images.nums[::3]:
        >>>     images.read(num)
        """
        return range(len(self.files))


# ----------------------------------------------------------------------------
# ============== Factory function to generate ImgSeries objects ==============
# ----------------------------------------------------------------------------


def series(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option.

    Parameters
    ----------
    cache : bool
        If True, use caching to keep images in memory once loaded.

    cache_size : int
        Maximum number of images kept in memory by the cache if used.

    *args
    **kwargs
        Any arrguments and keyword arguments accepted by ImgSeries.
    """
    if not cache:

        return ImgSeries(*args, **kwargs)

    else:

        class ImgSeriesCached(ImgSeries):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgSeriesCached(*args, **kwargs)
