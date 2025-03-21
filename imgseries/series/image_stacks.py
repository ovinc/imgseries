"""Class ImgSeries for image series manipulation"""

# Standard library imports
from pathlib import Path

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
        cache=False,
        Viewer=ImgSeriesViewer,
        Reader=None,
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
        self.path = Path(path)
        extension = self.path.suffix.lower()

        if Reader is None:
            if extension in ('.tif', '.tiff'):
                Reader = TiffStackReader
            elif extension == 'hdf5':
                Reader = HDF5Reader

        super().__init__(
            savepath=savepath,
            corrections=corrections,
            transforms=transforms,
            cache=cache,
            Viewer=Viewer,
            Reader=Reader,
        )

        self.data = self.reader.data
        self._get_initial_image_dims()

    def __repr__(self):
        return f"{super().__repr__()}\nfrom {self.path}"

    @property
    def nums(self):
        """Iterator (sliceable) of image identifiers.

        Examples
        --------
        Allows the user to do e.g.
        >>> for num in images.nums[::3]:
        >>>     images.read(num)
        """
        npts = self.reader.number_of_images
        return range(npts)

    @property
    def ntot(self):
        """Total number of images in the image series.

        Subclassed here car already available in image reader.

        Returns
        -------
        int
        """
        return self.reader.number_of_images
