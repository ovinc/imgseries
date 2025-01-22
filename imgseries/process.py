"""Misc. tools to process images"""

# Nonstandard
import skimage
import numpy as np


PIXEL_DEPTHS = {
    'uint8': 2**8 - 1,
    'uint16': 2**16 - 1,
}


def max_pixel_range(img):
    """Return max pixel value depending on img type, for use in plt.imshow.

    Parameters
    ----------
    img : array_like
       array containing the image data

    Returns
    -------
    tuple
        (vmin, vmax): max pixel value (None if not float or uint8/16)
    """
    dtype_name = img.dtype.name

    if 'float' in dtype_name:
        img_finite = img[np.isfinite(img)]  # remove nan and inf
        return img_finite.min(), img_finite.max()

    elif 'bool' in dtype_name:
        return 0, 1

    return 0, PIXEL_DEPTHS.get(img.dtype.name, None)


# =========== Define how to transform images (crop, rotate, etc.) ============


def divide(img, value):
    """Divide image by value, but keep initial data type

    Parameters
    ----------
    img : array_like
       array containing the image data

    value : float
        constant value that will be used to divide the pixel values

    Returns
    -------
    array_like
        processed image
    """
    # Avoids problems, e.g. np.uint8(257) is actually 1
    temp_img = np.clip(img / value, *max_pixel_range(img))
    return temp_img.astype(img.dtype)


def rgb_to_grey(img):
    """How to convert an RGB image to grayscale

    Parameters
    ----------
    img : array_like
       array containing the image data

    Returns
    -------
    array_like
        processed image
    """
    _, vmax = max_pixel_range(img)
    img_grey = skimage.color.rgb2gray(img)
    if type(vmax) is int:
        return (img_grey * vmax).astype(img.dtype)
    else:
        return img_grey


def double_threshold(img, vmin=None, vmax=None):
    """Threshold image (vmin <= v <= vmax --> 1, else 0).

    Parameters
    ----------
    img : array_like
       array containing the image data

    vmin : int or float
    vmax : int or float
        values for thresholding the image

    Returns
    -------
    array_like
        processed image

    Notes
    -----
        Returns binary (boolean) image (True / False).
    """
    if None in (vmin, vmax):
        val_min, val_max = max_pixel_range(img)
        vmin = val_min if vmin is None else vmin
        vmax = val_max if vmax is None else vmax
    condition = (img >= vmin) & (img <= vmax)
    return np.where(condition, True, False)


def gaussian_filter(img, size):
    """Gaussian filter to blur image

    Parameters
    ----------
    img : array_like
       array containing the image data

    size : float
        standard deviation (in pixels) of the gaussian filter

    Returns
    -------
    array_like
        processed image
    """
    _, vmax = max_pixel_range(img)
    img_filtered = skimage.filters.gaussian(img, sigma=size)
    if type(vmax) is int:
        return (img_filtered * vmax).astype(img.dtype)
    else:
        return img_filtered
