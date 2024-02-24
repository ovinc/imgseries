"""Class ImgSeries for image series manipulation"""

# Standard library imports
from functools import lru_cache

# Nonstandard
import filo

# local imports
from ..config import CONFIG, IMAGE_TRANSFORMS, IMAGE_CORRECTIONS
from ..managers import FileManager, ImageManager
from ..viewers import ImgSeriesViewer

from .general import ImgSeriesBase, ImageReader


class ImgSeriesReader(ImageReader):

    def _read(self, num):
        """read raw image from image series"""
        file = self.img_series.files[num].file
        return self.img_manager.read_image(file)


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
            viewer=viewer,
            img_manager=img_manager,
            file_manager=file_manager
        )

        self.img_reader = ImgSeriesReader(
            img_series=self,
            img_manager=img_manager,
        )

        self._get_initial_image_dims()

    def _set_substack(self, start, end, skip):
        """Generate subset of image numbers to be displayed/analyzed."""
        files = self.files[start:end:skip]
        nums = [file.num for file in files]
        return nums


# ----------------------------------------------------------------------------
# ============== Factory function to generate ImgSeries objects ==============
# ----------------------------------------------------------------------------


def series(*args, cache=False, cache_size=516, **kwargs):
    """Generator of ImgSeries object with a caching option."""
    if not cache:

        return ImgSeries(*args, **kwargs)

    else:

        class ImgSeriesCached(ImgSeries):

            cache = True

            @lru_cache(maxsize=cache_size)
            def read(self, num=0, transform=True, **kwargs):
                return super().read(num, transform=transform, **kwargs)

        return ImgSeriesCached(*args, **kwargs)
