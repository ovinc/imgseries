"""Class ImgSeries for image series manipulation"""

# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed

# Nonstandard
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm


class Export:
    """Export imgseries with transforms / display to files"""

    def __init__(
        self,
        img_series,
        filename='Img-',
        extension='.png',
        ndigits=5,
        folder='Export',
        with_display=False,
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

        with_display : bool
            if True, export images with display options as well
            (e.g., contrast, colormap, etc.)
        """
        self.img_series = img_series
        self.filename = filename
        self.extension = extension
        self.ndigits = ndigits
        self.export_folder = self.img_series.savepath / folder
        self.export_folder.mkdir(exist_ok=True)
        self.with_display = with_display

    def _run(self, num):
        """How to save individual image

        Parameters
        ----------
        num : int
            Image identifier in the image series
        """
        img = self.img_series.read(num=num)
        fname = f'{self.filename}{num:0{self.ndigits}}{self.extension}'
        filepath = self.export_folder / fname

        if self.with_display:
            kwargs = self.img_series._get_imshow_kwargs()
            plt.imsave(filepath, img, **kwargs)
        else:
            io.imsave(filepath, img)

    def run(
        self,
        start=0,
        end=None,
        skip=1,
        parallel=True,
    ):
        """Run the export process

        Parameters
        ----------
        start : int
        end : int
        skip : int
            images to consider. These numbers refer to 'num' identifier which
            starts at 0 in the first folder and can thus be different from the
            actual number in the image filename

        parallel : bool
            if True, use multiprocessing to save images faster
            (uses multiple cores of the computer)
        """
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
