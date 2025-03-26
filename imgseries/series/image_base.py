"""Class ImgSeries for image series manipulation"""

# Nonstandard
import matplotlib.pyplot as plt
from filo import DataSeries

# local imports
from ..config import CONFIG
from ..fileio import FileIO
from ..parameters.correction import CORRECTIONS
from ..parameters.transform import TRANSFORMS
from ..parameters.display import Display

from .line_profile import Profile
from .export import Export


class ImgSeriesBase(DataSeries):
    """Base class for series of images (or stacks)"""

    def __init__(
        self,
        savepath='.',
        corrections=None,
        transforms=None,
        cache=False,
        Reader=None,
        Viewer=None,
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
        corrs = CONFIG['correction order'] if corrections is None else corrections
        trans = CONFIG['transform order'] if transforms is None else transforms

        corrections_list = []
        for correction_name in corrs:
            cor = CORRECTIONS[correction_name](img_series=self)
            corrections_list.append(cor)

        transforms_list = []
        for transform_name in trans:
            tran = TRANSFORMS[transform_name](img_series=self)
            transforms_list.append(tran)

        reader = Reader(img_series=self, cache=cache)
        viewer = Viewer(img_series=self)

        super().__init__(
            savepath=savepath,
            corrections=corrections_list,
            transforms=transforms_list,
            reader=reader,
            viewer=viewer,
        )

        # Display options (do not impact analysis)
        self.display = Display(self)

    def _get_initial_image_dims(self):
        """Remember which type (B&W or color) and shape the raw images are"""
        img = self.read()
        self.initial_ny, self.initial_nx, *_ = img.shape
        self.initial_ndim = img.ndim

    @property
    def nx(self):
        """Image dimensions in the x-direction"""
        try:
            crop = self.crop
        except AttributeError:
            return self.initial_nx
        return self.initial_nx if crop.is_empty else self.crop.zone[2]

    @property
    def ny(self):
        """Image dimensions in the y-direction"""
        try:
            crop = self.crop
        except AttributeError:
            return self.initial_ny
        return self.initial_ny if crop.is_empty else self.crop.zone[3]

    @property
    def ndim(self):
        """Number of dimensions of the images (2 for gray level, 3 for color)"""
        try:
            grayscale = self.grayscale
        except AttributeError:
            return self.initial_ndim
        return self.initial_ndim if grayscale.is_empty else 2

    # ===================== Corrections and  Transforms ======================

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
        return super().load_transforms(filepath=self.savepath / (fname + '.json'))

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
        return super().save_transforms(filepath=self.savepath / (fname + '.json'))

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
        filepath = self.savepath / (fname + '.json')
        self.display.data = FileIO.from_json(filepath=filepath)

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
        filepath = self.savepath / (fname + '.json')
        FileIO.to_json(self.display.data, filepath=filepath)

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
        name = Transform.name

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
        name = Transform.name
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

    def profile(self, npts=100, radius=2, **kwargs):
        """Interactively get intensity profile by drawing a line on image."""
        profile = Profile(self, npts=npts, radius=radius, **kwargs)
        return profile

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
