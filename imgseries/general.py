"""Class ImgSeries for image series manipulation"""

# Standard library imports
import json
from pathlib import Path

# Nonstandard
import matplotlib.pyplot as plt
from skimage import io
import filo

# local imports
from .config import filenames
from .config import _read, _rgb_to_grey, _rotate, _crop
from .image_parameters import Rotation, Crop
from .plot import ImagePlot



class ImgSeriesPlot(ImagePlot):
    """See ImagePlot for details."""

    def create_plot(self):
        self.fig, self.ax = plt.subplots()

    def get_data(self, num):
        img = self.img_series.read(num, transform=self.transform)
        return {'num': num, 'image': img}

    def first_plot(self, data):

        if 'cmap' not in self.kwargs and data['image'].ndim < 3:
            self.kwargs['cmap'] = 'gray'

        self.imshow = self.ax.imshow(data['image'], **self.kwargs)

        self._display_info(data)
        self.ax.axis('off')

        self.updated_artists = [self.imshow]

    def update_plot(self, data):
        self.imshow.set_array(data['image'])
        self._display_info(data)

    def _display_info(self, data):

        num = data['num']

        if self.img_series.is_stack:
            title = 'Image'
        else:
            title = self.img_series.files[num].name

        raw_info = ' [RAW]' if not self.transform else ''

        self.ax.set_title(f'{title} (#{num}){raw_info}')



class ImgSeries(filo.Series):
    """Class to manage series of images, possibly in several folders."""

    # Only for __repr__ (str representation of class object, see filo.Series)
    name = 'Image Series'

    # Default filename to save file info with save_info (see filo.Series)
    info_filename = filenames['files'] + '.tsv'

    def __init__(self,
                 paths='.',
                 extension='.png',
                 savepath='.',
                 stack=None):
        """Init image series object.

        Parameters
        ----------
        - paths can be a string, path object, or a list of str/paths if data
          is stored in multiple folders.

        - extension: extension of files to consider (e.g. '.png')

        - savepath: folder in which to save parameter / analysis data.

        If file series is in a stack rather than in a series of images:
        - stack: path to the stack (.tiff) file
          (parameters paths & extension will be ignored)
        """
        # Image transforms that are applied to all images of the series.
        self.rotation = Rotation(self)
        self.crop = Crop(self)

        # Done here because self.stack will be an array, and bool(array)
        # generates warnings / errors
        self.is_stack = bool(stack)

        if self.is_stack:
            self.stack_path = Path(stack)
            self.stack = io.imread(stack, plugin="tifffile")
            self.savepath = Path(savepath)
        else:
            # Inherit useful methods and attributes for file series
            # (including self.savepath)
            super().__init__(paths=paths,
                             extension=extension,
                             savepath=savepath)

    def _rotate(self, img):
        """Rotate image according to pre-defined rotation parameters"""
        return _rotate(img, angle=self.rotation.data['angle'])

    def _crop(self, img):
        """Crop image according to pre-defined crop parameters"""
        return _crop(img, self.crop.data['zone'])

    def _from_json(self, filename):
        """"Load json file"""
        file = self.savepath / (filename + '.json')
        with open(file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    def _to_json(self, data, filename):
        """"Save data (dict) to json file"""
        file = self.savepath / (filename + '.json')
        with open(file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _set_substack(self, start, end, skip):
        """Generate subset of image numbers to be displayed/analyzed."""
        if self.is_stack:
            npts, *_ = self.stack.shape
            all_nums = list(range(npts))
            nums = all_nums[start:end:skip]
        else:
            files = self.files[start:end:skip]
            nums = [file.num for file in files]
        return nums

    def read(self, num=0, transform=True):
        """Load image data (image identifier num across folders).

        By default, if transforms are defined on the image (rotation, crop)
        then they are applied here. Put transform=False to only load the raw
        image in the stack.
        """
        if not self.is_stack:
            img = _read(self.files[num].file)
        else:
            img = self.stack[num]

        if transform and not self.rotation.is_empty:
            img = self._rotate(img)

        if transform and not self.crop.is_empty:
            img = self._crop(img)

        return img

    def load_transform(self, filename=None):
        """Return transform parameters (crop, rotation, etc.)
        from json file as a dictionary.

        If filename is not specified, use default filenames.

        If filename is specified, it must be an str without the extension, e.g.
        filename='Test' will load from Test.json.
        """
        self.rotation.reset()
        self.crop.reset()

        fname = filenames['transform'] if filename is None else filename
        transform_data = self._from_json(fname)

        self.rotation.data = transform_data['rotation']
        self.crop.data = transform_data['crop']

    def save_transform(self, filename=None):
        """Save transform parameters (crop, rotation etc.) into json file."""
        fname = filenames['transform'] if filename is None else filename
        transform_data = {'rotation': self.rotation.data,
                          'crop': self.crop.data}
        self._to_json(transform_data, fname)

    @staticmethod
    def rgb_to_grey(img):
        """"Convert RGB to grayscale"""
        return _rgb_to_grey(img)

    def show(self, num=0, transform=True, **kwargs):
        """Show image in a matplotlib window.

        Parameters
        ----------
        - num: image identifier in the file series
        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, load raw image.
        - **kwargs: matplotlib keyword arguments for ax.imshow()
        (note: cmap is grey by default for images with 1 color channel)
        """
        splot = ImgSeriesPlot(self, transform=transform, **kwargs)
        splot.create_plot()
        splot.plot(num=num)
        return splot.ax

    def inspect(self, start=0, end=None, skip=1, transform=True, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

        - kwargs: any keyword to pass to plt.imshow() (e.g. cmap='plasma')
        """
        nums = self._set_substack(start, end, skip)
        splot = ImgSeriesPlot(self, transform=transform, **kwargs)
        return splot.inspect(nums=nums)

    def animate(self, start=0, end=None, skip=1, transform=True, blit=False, **kwargs):
        """Interactively inspect image stack.

        Parameters:

        - start, end, skip: images to consider. These numbers refer to 'num'
          identifier which starts at 0 in the first folder and can thus be
          different from the actual number in the image filename

        - transform: if True (default), apply global rotation and crop (if defined)
                     if False, use raw images.

        - blit: if True, use blitting for faster animation.

        - kwargs: any keyword to pass to plt.imshow() (e.g. cmap='plasma')
        """
        nums = self._set_substack(start, end, skip)
        splot = ImgSeriesPlot(self, transform=transform, **kwargs)
        return splot.animate(nums=nums, blit=blit)
