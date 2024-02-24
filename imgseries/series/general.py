"""Class ImgSeries for image series manipulation"""

# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed

# Nonstandard
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

# local imports
from .line_profile import Profile
from ..config import CONFIG, IMAGE_TRANSFORMS, IMAGE_CORRECTIONS
from ..managers import FileManager, ImageManager
from ..viewers import ImgSeriesViewer
from ..parameters.transform import Transforms
from ..parameters.correction import Corrections
from ..parameters.display import Display


class ImageProcessor:
    """Base class for ImageTransformer, ImageCorrector and ImageReader"""

    def __init__(self, img_series, img_manager):
        """Parameters:

        - img_series: ImgSeries object
        - img_manager: ImageManager object
                       (can be customized when passed to ImgSeries)
        """
        self.img_series = img_series
        self.img_manager = img_manager


class ImageTransformer(ImageProcessor):
    """Class that connects ImageManager to an image series to act on it.

    (for global image transforms)
    """

    # NOTE: the name of the methods must correspond to the transform names
    # (parameter_type name in the transform)

    def rotation(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return self.img_manager.rotate(
            img=img,
            angle=self.img_series.rotation.data['angle'],
        )

    def crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return self.img_manager.crop(
            img=img,
            zone=self.img_series.crop.data['zone'],
        )

    def filter(self, img):
        """Filter / blur image according to pre-defined filter parameters"""
        return self.img_manager.filter(
            img=img,
            filter_type=self.img_series.filter.data['type'],
            size=self.img_series.filter.data['size'],
        )

    def subtraction(self, img):
        """Subtract pre-set reference image to current image."""
        img_ref = self.img_series.subtraction.reference_image
        return self.img_manager.subtract(
            img=img,
            img_ref=img_ref,
            relative=self.img_series.subtraction.relative,
        )

    def grayscale(self, img):
        """Convert RGB to grayscale"""
        return self.img_manager.rgb_to_grey(
            img=img,
        )

    def threshold(self, img):
        return self.img_manager.threshold(
            img=img,
            vmin=self.img_series.threshold.vmin,
            vmax=self.img_series.threshold.vmax,
        )


class ImageCorrector(ImageProcessor):
    """Class that connects ImageManager to an image series to act on it.

    (for image corrections)
    """

    # NOTE: the name of the methods must correspond to the correction names
    # (parameter_type name in the correction).

    def flicker(self, img, num):
        """Flicker correction by dividing image by factor"""
        return self.img_manager.divide(
            img=img,
            value=self.img_series.flicker.data['correction']['ratio'].loc[num]
        )

    def shaking(self, img, num):
        """NOT IMPLEMENTED YET // TODO"""
        return img


class ImageReader(ImageProcessor):
    """Class that connects ImageManager to an image series to act on it.

    (for reading images and applying transforms/corrections on them).
    This is a base class, children:
        - ImgSeriesReader
        - TiffStackReader
        - HDF5Reader (not implemented yet)
    """

    def apply_correction(self, img, num, correction_name):
        """Apply specific correction (str) to image and return new img array"""
        correction_object = getattr(self.img_series, correction_name)
        if correction_object.is_empty:
            return img
        correction_function = getattr(self.img_series.img_corrector, correction_name)
        return correction_function(img=img, num=num)

    def apply_corrections(self, img, num, **kwargs):
        """Apply stored corrections on the image (flicker, shaking, etc.)"""
        for correction_name in self.img_series.corrections:
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
        transform_object = getattr(self.img_series, transform_name)
        if transform_object.is_empty:
            return img
        transform_function = getattr(self.img_series.img_transformer, transform_name)
        return transform_function(img)

    def apply_transforms(self, img, **kwargs):
        """Apply stored transforms on the image (crop, rotation, etc.)"""
        for transform_name in self.img_series.transforms:
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
        corrections=IMAGE_CORRECTIONS,
        transforms=IMAGE_TRANSFORMS,
        viewer=ImgSeriesViewer,
        img_manager=ImageManager,
        file_manager=FileManager,
    ):
        """Init image series object.

        Parameters
        ----------
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
        self.file_manager = file_manager
        self.Viewer = viewer

        self.img_corrector = ImageCorrector(
            img_series=self,
            img_manager=img_manager,
        )

        self.img_transformer = ImageTransformer(
            img_series=self,
            img_manager=img_manager,
        )

        self.corrections = corrections
        for correction_name in self.corrections:
            correction_obj = Corrections[correction_name](img_series=self)
            # e.g. self.flicker = Flicker(self)
            setattr(self, correction_name, correction_obj)

        self.transforms = transforms
        for transform_name in self.transforms:
            transform_obj = Transforms[transform_name](img_series=self)
            # e.g. self.rotation = Rotation(self)
            setattr(self, transform_name, transform_obj)

        # Display options (do not impact analysis)
        self.display = Display(self)

    def _get_initial_image_dims(self):
        """Remember which type (B&W or color) and shape the raw images are"""
        img = self.read()
        self.ny, self.nx, *_ = img.shape
        self.initial_ndim = img.ndim
        self.ndim = self.initial_ndim

    # ===================== Corrections and  Transforms ======================

    @property
    def active_corrections(self):
        active_corrs = []
        for correction_name in self.corrections:
            correction_object = getattr(self, correction_name)
            if not correction_object.is_empty:
                active_corrs.append(correction_name)
        return active_corrs

    def reset_corrections(self):
        """Reset all active corrections."""
        for correction_name in self.active_corrections:
            correction_object = getattr(self, correction_name)
            correction_object.reset()

    @property
    def active_transforms(self):
        active_trnsfms = []
        for transform_name in self.transforms:
            transform_object = getattr(self, transform_name)
            if not transform_object.is_empty:
                active_trnsfms.append(transform_name)
        return active_trnsfms

    def reset_transforms(self):
        """Reset all active transforms."""
        for transform_name in self.active_transforms:
            transform_object = getattr(self, transform_name)
            transform_object.reset()

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
        transform_data = self.file_manager.from_json(self.savepath, fname)

        for transform_name in self.transforms:
            transform_object = getattr(self, transform_name)
            transform_object.data = transform_data.get(transform_name, {})
            transform_object._update_parameters()

    def save_transforms(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['transform'] if filename is None else filename
        transform_data = {}

        for transform_name in self.active_transforms:
            transform_object = getattr(self, transform_name)
            transform_data[transform_name] = transform_object.data

        self.file_manager.to_json(transform_data, self.savepath, fname)

    def load_display(self, filename=None):
        """Load display parameters (contrast, colormapn etc.) from json file.

        Display options are applied and stored in self.contrast, etc.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        self.display.data = self.file_manager.from_json(self.savepath, fname)

    def save_display(self, filename=None):
        """Save  display parameters (contrast, colormapn etc.) into json file.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        fname = CONFIG['filenames']['display'] if filename is None else filename
        self.file_manager.to_json(self.display.data, self.savepath, fname)

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
        nums = self._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.inspect(nums=nums)

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
        nums = self._set_substack(start, end, skip)
        viewer = self.Viewer(self, transform=transform, **kwargs)
        return viewer.animate(nums=nums, blit=blit)

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
          NOT IMPLEMENTED YET
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
        nums = self.img_series._set_substack(start, end, skip)

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
