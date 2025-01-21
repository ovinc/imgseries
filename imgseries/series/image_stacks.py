"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path
from functools import lru_cache

# local imports
from ..fileio import FileIO
from ..viewers import ImgSeriesViewer

from .general import ImgSeriesBase, ImageReader


class TiffStackReader(ImageReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._read_stack(file=self.img_series.path)

    @staticmethod
    def _read_stack(file):
        """I do it this way so that it's easier to subclass

        (User can provide own stack reader)
        """
        return FileIO.read_tiff_stack_whole(file=file)

    def _read(self, num):
        """read raw image from stack"""
        return self.data[num]

    @property
    def number_of_images(self):
        """number of images in the stack"""
        npts, *_ = self.data.shape
        return npts

    # This is an alternative method to avoid loading all in memory
    # BUT it's more difficult to get total number of images
    # def _read(self, num):
    #     """read raw image from stack"""
    #     return FileIO._read_tiff_stack_slice(file=self.img_series.path, num=num)


class HDF5Reader(ImageReader):
    """NOT IMPLEMENTED // TODO"""
    pass


class ImgStack(ImgSeriesBase):
    """Class to manage stacks of images (e.g., tiff, HDF5, etc.)"""

    is_stack = True

    def __init__(
        self,
        path,
        savepath='.',
        corrections=None,
        transforms=None,
        correction_order=None,
        transform_order=None,
        Viewer=ImgSeriesViewer,
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

        ImgReader : class
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
            correction_order=correction_order,
            transform_order=transform_order,
            Viewer=Viewer,
            ImgReader=ImgReader,
        )

        self.data = self.img_reader.data
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
        npts = self.img_reader.number_of_images
        return range(npts)

    @property
    def ntot(self):
        """Subclassed here car already available in image reader."""
        return self.img_reader.number_of_images


# ----------------------------------------------------------------------------
# ============== Factory function to generate ImgStack objects ===============
# ----------------------------------------------------------------------------


def stack(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option."""
    if not cache:

        return ImgStack(*args, **kwargs)

    else:

        class ImgStackCached(ImgStack):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgStackCached(*args, **kwargs)
