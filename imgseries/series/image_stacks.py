"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path
from functools import lru_cache

# local imports
from ..readers import HDF5Reader, TiffStackReader
from ..viewers import ImgSeriesViewer

from .image_base import ImgSeriesBase


class ImgStack(ImgSeriesBase):
    """Class to manage stacks of images (e.g., tiff, HDF5, etc.)"""

    is_stack = True

    def __init__(
        self,
        path,
        savepath='.',
        corrections=None,
        transforms=None,
        ImgViewer=ImgSeriesViewer,
        ImgReader=None,
    ):
        """Init image series object.

        Parameters
        ----------
        path : str or path object

        extension : str
            extension of files to consider (e.g. '.tiff')

        savepath : str or path object
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
        self.path = Path(path)
        self.savepath = Path(savepath)
        extension = self.path.suffix.lower()

        if ImgReader is None:
            if extension in ('.tif', '.tiff'):
                ImgReader = TiffStackReader
            elif extension == 'hdf5':
                ImgReader = HDF5Reader

        super().__init__(
            corrections=corrections,
            transforms=transforms,
            ImgViewer=ImgViewer,
            ImgReader=ImgReader,
        )

        self.data = self.img_reader.data
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
        npts = self.img_reader.number_of_images
        return range(npts)

    @property
    def ntot(self):
        """Total number of images in the image series.

        Subclassed here car already available in image reader.

        Returns
        -------
        int
        """
        return self.img_reader.number_of_images


# ----------------------------------------------------------------------------
# ============== Factory function to generate ImgStack objects ===============
# ----------------------------------------------------------------------------


def stack(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option.

    Parameters
    ----------
    cache : bool
        If True, use caching to keep images in memory once loaded.

    cache_size : int
        Maximum number of images kept in memory by the cache if used.

    *args
    **kwargs
        Any arrguments and keyword arguments accepted by ImgStack.
    """
    if not cache:

        return ImgStack(*args, **kwargs)

    else:

        class ImgStackCached(ImgStack):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgStackCached(*args, **kwargs)
