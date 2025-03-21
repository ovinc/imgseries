"""Class ImgSeries for image series manipulation"""

# Nonstandard
import filo

# local imports
from ..config import CONFIG
from ..readers import SingleImageReader
from ..viewers import ImgSeriesViewer

from .image_base import ImgSeriesBase


class ImgSeries(ImgSeriesBase):
    """Class to manage series of images, possibly in several folders."""

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = CONFIG['filenames']['files'] + '.tsv'

    # To distinguish between ImgSeries and ImgStack
    is_stack = False

    def __init__(
        self,
        folders='.',
        extension='.png',
        savepath='.',
        corrections=None,
        transforms=None,
        cache=False,
        Viewer=ImgSeriesViewer,
        Reader=SingleImageReader,
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

        cache : bool
            if True, use caching for speed improvement
            (both for loading files and transforms)
            this is useful when calling read() multiple times on the same
            image (e.g. when inspecting series/stacks)

        Viewer : subclass of ImageViewerBase
            which Viewer class to use for show(), inspect() etc.
            if None, use default viewer class

        Reader : subclass of ImageReaderBase
            class (or object) that defines how to read images
        """
        self.files = filo.FileSeries.auto(
            folders=folders,
            extension=extension,
            refpath=savepath,
        )

        super().__init__(
            savepath=savepath,
            corrections=corrections,
            transforms=transforms,
            cache=cache,
            Viewer=Viewer,
            Reader=Reader,
        )

        self._get_initial_image_dims()

    def __repr__(self):
        return f"{super().__repr__()}\nfrom {self.files}"

    @property
    def info(self):
        return self.files.info

    def _get_info_filepath(self, filename):
        filename = CONFIG['filenames']['files'] if filename is None else filename
        return self.savepath / filename

    def load_times(self, filename=None):
        self.files.update_times(
            filepath=self._get_info_filepath(filename),
            sep=CONFIG['csv separator'],
        )

    def save_info(self, filename=None):
        self.files.to_csv(
            filepath=self._get_info_filepath(filename),
            sep=CONFIG['csv separator'],
        )

    @property
    def nums(self):
        """Iterator (sliceable) of image identifiers.

        Examples
        --------
        Allows the user to do e.g.
        >>> for num in images.nums[::3]:
        >>>     images.read(num)
        """
        return range(self.ntot)

    @property
    def ntot(self):
        """Total number of image files in the series.

        Returns
        -------
        int
        """
        return len(list(self.files))
