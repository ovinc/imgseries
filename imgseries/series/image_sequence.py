"""ImgSequence, using array structures as inputs (e.g. pims image sequences)"""

# local imports
from ..viewers import ImgSeriesViewer
from .image_base import ImgSeriesBase
from ..readers import ImgSequenceReader


class ImgSequence(ImgSeriesBase):
    """Class to manage img sequences from array-like sequences (e.g. pims)"""

    # Is condidered a stack because no individual files for each image
    is_stack = True

    def __init__(
        self,
        img_sequence,
        savepath='.',
        corrections=None,
        transforms=None,
        cache=False,
        Viewer=None,
        Reader=None,
    ):
        """Init image series object.

        Parameters
        ----------
        img_sequence : array-like
            pims.ImageSequence object or equivalent

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
        """
        self.img_sequence = img_sequence

        try:
            self.path = self.img_sequence.pathname   # defined by pims
        except AttributeError:
            self.path = savepath  # useful only for analysis metadata

        super().__init__(
            savepath=savepath,
            corrections=corrections,
            transforms=transforms,
            cache=cache,
            Viewer=ImgSeriesViewer if Viewer is None else Viewer,
            Reader=ImgSequenceReader if Reader is None else Reader,
        )
        self._get_initial_image_dims()

    def __repr__(self):
        return f"{super().__repr__()}\nfrom {self.img_sequence}"

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
        """Total number of images in the image series.

        Subclassed here car already available in image reader.

        Returns
        -------
        int
        """
        return len(self.img_sequence)
