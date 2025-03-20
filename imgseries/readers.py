"""Class ImgSeries for image series manipulation"""

# Non-standard
from filo import DataSeriesReaderBase

# local imports
from .config import CONFIG
from .fileio import FileIO


class ImageReaderBase(DataSeriesReaderBase):
    """Base class for reading images.

    (for reading images and applying transforms/corrections on them).
    This is a base class, children:
        - SingleImageReader
        - TiffStackReader
        - HDF5Reader (not implemented yet)
    """
    def __init__(self, img_series, cache=False):
        """Init Image reader

        Parameters
        ----------

        img_series : ImgSeries object
        cache : bool
        """
        super().__init__(
            data_series=img_series,
            cache=cache,
            read_cache_size=CONFIG['read cache size'],
            transform_cache_size=CONFIG['transform cache size'],
        )
        self.img_series = img_series


# ============================= Children classes =============================


class SingleImageReader(ImageReaderBase):

    @staticmethod
    def _read_image(filepath):
        """Load image array from file

        Parameters
        ----------
        filepath : str or pathlib.Path
            file to load the image data from

        Returns
        -------
        array-like
            image as an array (typically np.array)

        Notes
        -----
            This is here for customization
            (can be subclassed to use other reading method)
        """
        return FileIO.read_single_image(filepath=filepath)

    def _read(self, num):
        """read raw image from image series"""
        filepath = self.img_series.files[num].path
        return self._read_image(filepath=filepath)


class TiffStackReader(ImageReaderBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._read_stack(filepath=self.img_series.path)

    @staticmethod
    def _read_stack(filepath):
        """Read whole stack of images into memory (to be done on init)

        I do it this way so that it's easier to subclass
        (User can provide own stack reader)
        """
        return FileIO.read_tiff_stack_whole(filepath=filepath)

    def _read(self, num):
        """read single image (slice) from stack"""
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
    #     return FileIO._read_tiff_stack_slice(filepath=self.img_series.path, num=num)


class HDF5Reader(ImageReaderBase):
    """NOT IMPLEMENTED // TODO"""
    pass
