"""Class ImgSeries for image series manipulation"""

# Nonstandard
import matplotlib.pyplot as plt

# local imports
from ..config import CONFIG
from ..fileio import FileIO
from ..parameters.correction import CORRECTIONS
from ..parameters.transform import TRANSFORMS
from ..parameters.display import Display

from .line_profile import Profile
from .export import Export


class ImgSeriesBase:
    """Base class for series of images (or stacks)"""

    # cache images during read() or not
    cache = False

    def __init__(
        self,
        corrections=None,
        transforms=None,
        ImgViewer=None,
        ImgReader=None,
    ):
        """Init image series object.

        Parameters
        ----------
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
        self.ImgViewer = ImgViewer
        self.img_reader = ImgReader(self)

        self.corrections = CONFIG['correction order'] if corrections is None else corrections
        self.transforms = CONFIG['transform order'] if transforms is None else transforms

        for correction_name in self.corrections:
            correction = CORRECTIONS[correction_name](img_series=self)
            # e.g. self.flicker = Flicker(self)
            setattr(self, correction_name, correction)

        for transform_name in self.transforms:
            transform = TRANSFORMS[transform_name](img_series=self)
            # e.g. self.rotation = Rotation(self)
            setattr(self, transform_name, transform)

        # Display options (do not impact analysis)
        self.display = Display(self)

    def _get_initial_image_dims(self):
        """Remember which type (B&W or color) and shape the raw images are"""
        img = self.read()
        self.ny, self.nx, *_ = img.shape
        self.initial_ndim = img.ndim
        self.ndim = self.initial_ndim

    # =========================== Iteration tools ============================

    @property
    def nums(self):
        """Iterator (sliceable) of image identifiers.

        Define in subclasses.

        Examples
        --------
        Allows the user to do e.g.
        >>> for num in images.nums[::3]:
        >>>     images.read(num)
        """
        pass

    @property
    def ntot(self):
        """Total number of images in the image series.

        Can be subclassed if necessary.

        Returns
        -------
        int
        """
        return len(self.nums)

    # ===================== Corrections and  Transforms ======================

    @property
    def active_corrections(self):
        active_corrs = []
        for correction_name in self.corrections:
            correction = getattr(self, correction_name)
            if not correction.is_empty:
                active_corrs.append(correction_name)
        return active_corrs

    def reset_corrections(self):
        """Reset all active corrections."""
        for correction_name in self.active_corrections:
            correction = getattr(self, correction_name)
            correction.reset()

    @property
    def active_transforms(self):
        active_trnsfms = []
        for transform_name in self.transforms:
            transform = getattr(self, transform_name)
            if not transform.is_empty:
                active_trnsfms.append(transform_name)
        return active_trnsfms

    def reset_transforms(self):
        """Reset all active transforms."""
        for transform_name in self.active_transforms:
            transform = getattr(self, transform_name)
            transform.reset()

    @staticmethod
    def add_transform(Transform, order=None):
        """Add custom transform to the available transforms

        Parameters
        ----------
        Transform : subclass of TransformParameter
            transform class created by user.
        order : int
            at which position in the transform sequence to apply transform
            if None (default), add at the end of the transform list.
        """
        name = Transform.parameter_name

        try:
            TRANSFORMS[name]
        except KeyError:  # Does not exist yet --> OK
            pass
        else:
            raise ValueError(
                f"'{name}' already exists as a transform name. "
                'Please use another name, or remove the existing transform'
                'with the same name.'
            )

        TRANSFORMS[name] = Transform

        old_order = CONFIG['transform order']
        if order is None:
            new_order = old_order + (name,)
        else:
            new_order = old_order[:order] + (name,) + old_order[order:]
        CONFIG['transform order'] = new_order

    @staticmethod
    def remove_transform(Transform):
        """Remove custom transform from the available transforms

        Parameters
        ----------
        Transform : subclass of TransformParameter
            transform class created by user.
        """
        name = Transform.parameter_name
        current_order = CONFIG['transform order']

        try:
            TRANSFORMS.pop(name)
        except KeyError:
            raise ValueError(
                f"'{name}' transform does not exist. "
                f"Existing transforms: [{', '.join(current_order)}]"
            )

        new_order = tuple(n for n in current_order if n != name)
        CONFIG['transform order'] = new_order

    # ============================= Misc. tools ==============================

    def _get_imshow_kwargs(self, transform=True):
        """Define kwargs to pass to imshow (to have grey by default for 2D)."""
        kwargs = {**self.display.data}
        if self.ndim < 3:
            kwargs['cmap'] = kwargs.get('cmap', 'gray')
        # Without lines below, display of thresholded images is buggy when
        # contrast has been defined previously
        if transform and not self.threshold.is_empty:
            kwargs['vmin'] = 0
            kwargs['vmax'] = 1
        return kwargs

    def _imshow(self, img, ax=None, transform=True, **kwargs):
        """Use plt.imshow() with default kwargs and/or additional ones

        Parameters
        ----------
        img : array_like
            image to display

        ax : plt.Axes
            axes in which to display the image. If not specified (None),
            create new ones

        transform : bool
            whether to apply transforms or show raw image.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)

        Returns
        -------
        plt.Axes.imshow() object
        """
        if ax is None:
            _, ax = plt.subplots()
        default_kwargs = self._get_imshow_kwargs(transform=transform)
        kwargs = {**default_kwargs, **kwargs}
        return ax.imshow(img, **kwargs)

    # ============================ Public methods ============================

    def read(self, num=0, correction=True, transform=True, **kwargs):
        """Load image data as an array.

        Parameters
        ----------
        num : int
            image identifier

        correction : bool
            By default, if corrections are defined on the image
            (flicker, shaking etc.), then they are applied here.
            Put correction=False to only load the raw image in the stack.

        transform : bool
            By default, if transforms are defined on the image
            (rotation, crop etc.), then they are applied here.
            Put transform=False to only load the raw image in the stack.

        **kwargs
            by default if transform=True, all active transforms are applied.
            Set any transform name to False to not apply this transform.
            e.g. images.read(subtraction=False).

        Returns
        -------
        array_like
            Image data as an array
        """
        return self.img_reader.read(
            num=num,
            correction=correction,
            transform=transform,
            **kwargs,
        )

    def profile(self, npts=100, radius=2, **kwargs):
        """Interactively get intensity profile by drawing a line on image."""
        profile = Profile(self, npts=npts, radius=radius, **kwargs)
        return profile

    def load_transforms(self, filename=None):
        """Load transform parameters (crop, rotation, etc.) from json file.

        Parameters
        ----------
        filename : str

            If filename is not specified (None), use default filenames.

            If filename is specified, it must be an str without the extension,
            e.g. filename='Test' will load from Test.json.

        Returns
        -------
        None
            But transforms are applied and stored in attributes, e.g.
            self.rotation, self.crop, etc.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        file = self.savepath / (fname + '.json')
        transform_data = FileIO.from_json(file=file)

        for transform_name in self.transforms:
            transform = getattr(self, transform_name)
            transform.data = transform_data.get(transform_name, {})
            transform._update_parameters()

    def save_transforms(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file.

        Parameters
        ----------
        filename : str

            If filename is not specified (None), use default filenames.

            If filename is specified, it must be an str without the extension,
            e.g. filename='Test' will save to Test.json.

        Returns
        -------
        None
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        transform_data = {}

        for transform_name in self.active_transforms:
            transform = getattr(self, transform_name)
            transform_data[transform_name] = transform.data

        file = self.savepath / (fname + '.json')
        FileIO.to_json(transform_data, file=file)

    def load_display(self, filename=None):
        """Load display parameters (contrast, colormapn etc.) from json file.

        Parameters
        ----------
        filename : str

            If filename is not specified (None), use default filenames.

            If filename is specified, it must be an str without the extension,
            e.g. filename='Test' will load from Test.json.

        Returns
        -------
        None
            But display option are applied and stored in attributes, e.g.
            self.contrast, etc.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        file = self.savepath / (fname + '.json')
        self.display.data = FileIO.from_json(file=file)

    def save_display(self, filename=None):
        """Save  display parameters (contrast, colormapn etc.) into json file.

        Parameters
        ----------
        filename : str

            If filename is not specified (None), use default filenames.

            If filename is specified, it must be an str without the extension,
            e.g. filename='Test' will save to Test.json.

        Returns
        -------
        None
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        file = self.savepath / (fname + '.json')
        FileIO.to_json(self.display.data, file=file)

    # ==================== Interactive inspection methods ====================

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        num : int
            image identifier in the file series

        transform : bool
            if True (default), apply active transforms
            if False, load raw image.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        viewer = self.ImgViewer(self, transform=transform, **kwargs)
        return viewer.show(num=num)

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image series.

        Parameters
        ----------
        start : int
        end : int
        skip : int
            images to consider. These numbers refer to 'num' identifier which
            starts at 0 in the first folder and can thus be different from the
            actual number in the image filename

        transform : bool
            if True (default), apply active transforms
            if False, use raw images.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        viewer = self.ImgViewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=self.nums[start:end:skip])

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters
        ----------
        start : int
        end : int
        skip : int
            images to consider. These numbers refer to 'num' identifier which
            starts at 0 in the first folder and can thus be different from the
            actual number in the image filename

        transform : bool
            if True (default), apply active transforms
            if False, use raw images.

        **kwargs
            any keyword-argument to pass to imshow() (overrides default
            and preset display parameters such as contrast, colormap etc.)
            (note: cmap is grey by default for 2D images)
        """
        viewer = self.ImgViewer(self, transform=transform, **kwargs)
        return viewer.animate(nums=self.nums[start:end:skip], blit=blit)

    # =========================== Export to files ============================

    def export(
        self,
        filename='Img-',
        extension='.png',
        ndigits=5,
        folder='Export',
        start=0,
        end=None,
        skip=1,
        with_display=False,
        parallel=True,
    ):
        """Export images to files

        Parameters
        ----------
        filename : str
            Base name for the image files

        extension : str
            Extension of the images (e.g. '.png')

        ndigits : int
            Number of digits to use to display image number in file name

        folder : str
            Name of directory in which images will be saved

        start : int
        end : int
        skip : int
            images to consider. These numbers refer to 'num' identifier which
            starts at 0 in the first folder and can thus be different from the
            actual number in the image filename

        with_display : bool
            if True, export images with display options as well
            (e.g., contrast, colormap, etc.)

        parallel : bool
            if True, use multiprocessing to save images faster
            (uses multiple cores of the computer)
        """
        export = Export(
            self,
            filename=filename,
            extension=extension,
            ndigits=ndigits,
            folder=folder,
            with_display=with_display,
        )

        export.run(
            start=start,
            end=end,
            skip=skip,
            parallel=parallel,
        )
