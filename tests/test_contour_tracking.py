"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See ExamplesSeries.ipynb
Jupyter notebook for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import series, stack
from imgseries import ContourTracking, ContourTrackingResults


# =============================== Misc. config ===============================

modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'data/for-tests-do-not-modify'
folders = [basefolder / '..' / folder for folder in ('img1', 'img2')]

tiff_stack = Path('data/stack') / 'ImgStack.tif'

# ================== Test contour tracking on image series ===================

images = series(folders, savepath=basefolder)
images.load_time('Img_Files_Saved.tsv')  # in case files have changed creation time

ct = ContourTracking(images)
ct.contours.load('Img_ContourTracking_Saved')


def test_contour_tracking_basic():
    ct.run()
    assert len(ct.results.data) == 50


def test_contour_tracking_range():
    ct.run(start=10, skip=3)
    assert ct.results.data.shape == (14, 15)


def test_contour_tracking_load():
    ctresults = ContourTrackingResults(savepath=basefolder)
    ctresults.load('Img_ContourTracking_Saved')
    assert round(ctresults.data.at[4, 'x3']) == 418


# ================== Test contour tracking on image stacks ===================

img_stack = stack(tiff_stack)

ctstack = ContourTracking(img_stack, savepath=basefolder / 'stack')
ctstack.contours.load('Img_ContourTracking_Saved')


def test_contourstack_tracking_basic():
    ctstack.run()
    assert ctstack.img_series.data.shape == (200, 100, 112)
    assert len(ctstack.results.data) == 200
