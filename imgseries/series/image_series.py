"""Class ImgSeries for image series manipulation"""

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
        cache=False,
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

        cache : bool
            if True, use caching for speed improvement
            (both for loading files and transforms)
            this is useful when calling read() multiple times on the same
            image (e.g. when inspecting series/stacks)

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
            cache=cache,
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
