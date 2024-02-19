"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path
from functools import lru_cache

# local imports
from ..config import IMAGE_TRANSFORMS, IMAGE_CORRECTIONS
from ..managers import FileManager, ImageManager
from ..viewers import ImgSeriesViewer

from .general import ImgSeriesBase, ImageReader


class TiffStackReader(ImageReader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.img_manager.read_tiff_stack_whole(
            file=self.img_series.path,
        )

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
    #     return self.img_manager.read_tiff_stack_slice(
    #         file=self.img_series.path,
    #         num=num
    #     )


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
        corrections=IMAGE_CORRECTIONS,
        transforms=IMAGE_TRANSFORMS,
        viewer=ImgSeriesViewer,
        img_manager=ImageManager,
        file_manager=FileManager,
    ):
        """Init image series object.

        Parameters
        ----------
        - paths can be a string, path object, or a list of str/paths if data
          is stored in multiple folders.

        - extension: extension of files to consider (e.g. '.png')

        - savepath: folder in which to save parameters (transform, display etc.)

        - corrections: iterable of name of corrections to consider (their
                       order indicates the order in which they are applied),
                       e.g. corrections=('shaking', 'flicker')

        - transforms: iterable of names of transforms to consider (their order
                      indicates the order in which they are applied), e.g.
                      transforms=('rotation', 'crop', 'filter')

        - viewer: which Viewer class to use for show(), inspect() etc.

        - img_manager: class (or object) that defines how to read and
                       transform images

        - file_manager: class (or object) that defines how to interact with
                        saved files
        """
        super().__init__(
            corrections=corrections,
            transforms=transforms,
            viewer=viewer,
            img_manager=img_manager,
            file_manager=file_manager
        )

        self.path = Path(path)
        self.savepath = Path(savepath)
        extension = self.path.suffix.lower()

        if extension in ('.tif', '.tiff'):
            StackReader = TiffStackReader
        elif extension == 'hdf5':
            StackReader = HDF5Reader

        self.img_reader = StackReader(
            img_series=self,
            img_manager=img_manager,
        )

        self.data = self.img_reader.data

        self._get_initial_image_dims()

    def _set_substack(self, start, end, skip):
        """Generate subset of image numbers to be displayed/analyzed."""
        npts = self.img_reader.number_of_images
        all_nums = list(range(npts))
        nums = all_nums[start:end:skip]
        return nums


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
