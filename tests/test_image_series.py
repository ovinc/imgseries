"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See example
Jupyter notebooks for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import ImgSeries, ImgStack


# =============================== Misc. config ===============================

modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'examples/data/for-tests-do-not-modify'
folders = [basefolder / '..' / folder for folder in ('img1', 'img2')]

images = ImgSeries(folders, savepath=basefolder)
images.load_times('Img_Files.tsv')  # in case files have changed creation time

tiff_stack = Path('examples/data/stack') / 'ImgStack.tif'
img_stack = ImgStack(tiff_stack)

# ======================== Test general image series =========================


def test_set_global_transform():
    """Setting a global transform (rotation, crop) on image ImgSeries"""
    n = 22

    images.rotation.reset()
    images.crop.reset()

    images.rotation.angle = -66.6
    img_raw = images.read(n, transform=False)
    img_rot = images.read(n)

    images.crop.zone = (186, 193, 391, 500)
    img_crop = images.read(n)

    # Expand the tests below:
    images.filter.size = 8
    images.subtraction.reference = (1, 2)
    images.subtraction.relative = True
    images.threshold.vmin = 0.003

    assert img_raw.shape == (550, 608)
    assert img_rot.shape == (776, 746)
    assert img_crop.shape == (500, 391)


def test_load_global_transform():
    """Loading a saved global transform (rotation, crop)"""
    n = 11

    images.rotation.reset()
    images.crop.reset()
    images.load_transforms()

    img_raw = images.read(n, transform=False)
    img_tot = images.read(n)

    assert img_raw.shape == (550, 608)
    assert img_tot.shape == (380, 467)


def test_img_time():
    """General test of setting and getting image times."""
    n = 33
    assert round(images.info.loc[n, 'time (unix)']) == 1599832463
    assert images.info.loc[n, 'folder'] in ['../img2', '..\\img2']
    assert images.info.loc[n, 'filename'] == 'img-00643.png'


def test_img_time_update():
    """Test loading external time data."""
    n = 40
    images.load_times('Img_Files_Rounded.tsv')
    assert images.info.loc[n, 'time (unix)'] == 1599832477
    images.load_times('Img_Files.tsv')  # resset to previous state


def test_read_stack():
    """Read data from ImgStack file"""
    img = img_stack.read(num=10)
    assert img.shape == (100, 112)


def test_slicing_images():
    """Test looping using slicing of nums"""
    for num in images.nums[::20]:
        val_max = images.read(num).max()
    assert val_max > 0


def test_slicing_stack():
    """Test looping using slicing of nums"""
    for num in img_stack.nums[::20]:
        val_max = img_stack.read(num).max()
    assert val_max > 0
