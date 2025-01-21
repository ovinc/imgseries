"""Class ImgSeries for image series manipulation"""

# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed

# Nonstandard
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

# local imports
from .line_profile import Profile
from ..config import CONFIG
from ..fileio import FileIO
from ..viewers import ImgSeriesViewer
from ..parameters.transform import TRANSFORMS
from ..parameters.correction import CORRECTIONS
from ..parameters.display import Display


class ImageReader:
    """Class that connects ImageManager to an image series to act on it.

    (for reading images and applying transforms/corrections on them).
    This is a base class, children:
        - ImgSeriesReader
        - TiffStackReader
        - HDF5Reader (not implemented yet)
    """
    def __init__(self, img_series):
        """Parameters:

        - img_series: ImgSeries object
        """
        self.img_series = img_series

    def apply_correction(self, img, num, correction_name):
        """Apply specific correction (str) to image and return new img array"""
        correction = getattr(self.img_series, correction_name)
        if correction.is_empty:
            return img
        return correction.apply(img=img, num=num)

    def apply_corrections(self, img, num, **kwargs):
        """Apply stored corrections on the image (flicker, shaking, etc.)"""
        for correction_name in self.img_series.correction_order:
            # Do not consider any correction specifically marked as false
            if kwargs.get(correction_name, True):
                img = self.apply_correction(
                    img=img,
                    num=num,
                    correction_name=correction_name,
                )
        return img

    def apply_transform(self, img, transform_name):
        """Apply specific transform (str) to image and return new img array"""
        transform = getattr(self.img_series, transform_name)
        if transform.is_empty:
            return img
        return transform.apply(img)

    def apply_transforms(self, img, **kwargs):
        """Apply stored transforms on the image (crop, rotation, etc.)"""
        for transform_name in self.img_series.transform_order:
            # Do not consider any transform specifically marked as false
            if kwargs.get(transform_name, True):
                img = self.apply_transform(
                    img=img,
                    transform_name=transform_name,
                )
        return img

    def _read(self, num):
        """How to read image from series/stack. To be defined in subclasses"""
        pass

    def read(self, num, correction=True, transform=True, **kwargs):
        """Read image #num in image series and apply transforms if requested.

        Kwargs can be rotation=True or threshold=False to switch on/off
        transforms during the processing of the image
        """
        img = self._read(num=num)
        img = self.apply_corrections(img, num, **kwargs) if correction else img
        img = self.apply_transforms(img, **kwargs) if transform else img
        return img


# ========================= MAIN IMAGE SERIES CLASS ==========================


class ImgSeriesBase:
    """Base class for series of images (or stacks)"""

    # cache images during read() or not
    cache = False

    def __init__(
        self,
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
        self.Viewer = Viewer
        self.img_reader = ImgReader(self)

        Corrections = CORRECTIONS if corrections is None else corrections
        Transforms = TRANSFORMS if transforms is None else transforms

        self.correction_order = CONFIG['correction order'] if correction_order is None else correction_order
        self.transform_order = CONFIG['transform order'] if transform_order is None else transform_order

        for correction_name in self.correction_order:
            correction = Corrections[correction_name](img_series=self)
            # e.g. self.flicker = Flicker(self)
            setattr(self, correction_name, correction)

        for transform_name in self.transform_order:
            transform = Transforms[transform_name](img_series=self)
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

        Allows the user to do e.g.
        ```python
        for num in images.nums[::3]:
            images.read(num)
        ```
        Define in subclasses.
        """
        pass

    @property
    def ntot(self):
        """Total number of images in the image series.

        Can be subclassed if necessary.
        """
        return len(self.nums)

    # ===================== Corrections and  Transforms ======================

    @property
    def active_corrections(self):
        active_corrs = []
        for correction_name in self.correction_order:
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
        for transform_name in self.transform_order:
            transform = getattr(self, transform_name)
            if not transform.is_empty:
                active_trnsfms.append(transform_name)
        return active_trnsfms

    def reset_transforms(self):
        """Reset all active transforms."""
        for transform_name in self.active_transforms:
            transform = getattr(self, transform_name)
            transform.reset()

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
        - img: image to display (numpy array or equivalent)

        - ax: axes in which to display the image. If not specified, create new
              ones

        - transform: whether to apply transforms or show raw image.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
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

        - num: image identifier (integer)

        - transform: By default, if corrections are defined on the image
                     (flicker, shaking etc.), then they are applied here.
                     Put correction=False to only load the raw image in the
                     stack.

        - transform: By default, if transforms are defined on the image
                     (rotation, crop etc.), then they are applied here.
                     Put transform=False to only load the raw image in the
                     stack.

        - kwargs: by default if transform=True, all active transforms are
                  applied. Set any transform name to False to not apply
                  this particular transform.
                  e.g. images.read(subtraction=False)
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

        Transforms are applied and stored in self.rotation, self.crop, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        transform_data = FileIO.from_json(self.savepath, fname)

        for transform_name in self.transform_order:
            transform = getattr(self, transform_name)
            transform.data = transform_data.get(transform_name, {})
            transform._update_parameters()

    def save_transforms(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        transform_data = {}

        for transform_name in self.active_transforms:
            transform = getattr(self, transform_name)
            transform_data[transform_name] = transform.data

        FileIO.to_json(transform_data, self.savepath, fname)

    def load_display(self, filename=None):
        """Load display parameters (contrast, colormapn etc.) from json file.

        Display options are applied and stored in self.contrast, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        self.display.data = FileIO.from_json(self.savepath, fname)

    def save_display(self, filename=None):
        """Save  display parameters (contrast, colormapn etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        FileIO.to_json(self.display.data, self.savepath, fname)

    # ==================== Interactive inspection methods ====================

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series

        - transform: if True (default), apply active transforms
                     if False, load raw image.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.show(num=num)

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image series.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=self.nums[start:end:skip])

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply active transforms
                     if False, use raw images.

        - blit: if True, use blitting for faster animation.

        - kwargs: any keyword-argument to pass to imshow() (overrides default
          and preset display parameters such as contrast, colormap etc.)
          (note: cmap is grey by default for 2D images)
        """
        viewer = self.Viewer(self, transform=transform, **kwargs)
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
        """Export images

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - with_display: if True, export images with display options as well
          (e.g., contrast, colormap, etc.)
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


class Export:

    def __init__(
        self,
        img_series,
        filename='Img-',
        extension='.png',
        ndigits=5,
        folder='Export',
        with_display=False,
    ):
        """Export images."""
        self.img_series = img_series
        self.filename = filename
        self.extension = extension
        self.ndigits = ndigits
        self.export_folder = self.img_series.savepath / folder
        self.export_folder.mkdir(exist_ok=True)
        self.with_display = with_display

    def _run(self, num):
        img = self.img_series.read(num=num)
        fname = f'{self.filename}{num:0{self.ndigits}}{self.extension}'
        file = self.export_folder / fname

        if self.with_display:
            kwargs = self.img_series._get_imshow_kwargs()
            plt.imsave(file, img, **kwargs)
        else:
            io.imsave(file, img)

    def run(
        self,
        start=0,
        end=None,
        skip=1,
        parallel=True,
    ):
        nums = self.img_series.nums[start:end:skip]

        if not parallel:
            for num in tqdm(nums):
                self._run(num)
            return

        futures = {}

        with ProcessPoolExecutor() as executor:

            for num in nums:
                future = executor.submit(self._run, num)
                futures[num] = future

            # Waitbar ----------------------------------------------------
            futures_list = list(futures.values())
            for future in tqdm(as_completed(futures_list), total=len(nums)):
                pass
