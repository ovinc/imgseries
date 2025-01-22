"""Class ImgSeries for image series manipulation"""

# local imports
from .fileio import FileIO


class ImageReaderBase:
    """Base class for reading images.

    (for reading images and applying transforms/corrections on them).
    This is a base class, children:
        - SingleImageReader
        - TiffStackReader
        - HDF5Reader (not implemented yet)
    """
    def __init__(self, img_series):
        """Parameters:

        img_series : ImgSeries object
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
        transform = getattr(self.img_series, transform_name)
        if transform.is_empty:
            return img
        return transform.apply(img)

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


# ============================= Children classes =============================


class SingleImageReader(ImageReaderBase):

    @staticmethod
    def _read_image(file):
        return FileIO.read_single_image(file=file)

    def _read(self, num):
        """read raw image from image series"""
        file = self.img_series.files[num].file
        return self._read_image(file)


class TiffStackReader(ImageReaderBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self._read_stack(file=self.img_series.path)

    @staticmethod
    def _read_stack(file):
        """Read whole stack of images into memory (to be done on init)

        I do it this way so that it's easier to subclass
        (User can provide own stack reader)
        """
        return FileIO.read_tiff_stack_whole(file=file)

    def _read(self, num):
        """read single image (slice) from stack"""
        return self.data[num]

    @property
    def number_of_images(self):
        """number of images in the stack"""
        npts, *_ = self.data.shape
        return npts

    # This is an alternative method to avoid loading all in memory
    # BUT it's more difficult to get total number of images
    # def _read(self, num):
    #     """read raw image from stack"""
    #     return FileIO._read_tiff_stack_slice(file=self.img_series.path, num=num)


class HDF5Reader(ImageReaderBase):
    """NOT IMPLEMENTED // TODO"""
    pass
