"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See example
Jupyter notebooks for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import ImgSeries, ImgStack
from imgseries import ContourTracking, ContourTrackingResults


# =============================== Misc. config ===============================

modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'examples/data/for-tests-do-not-modify'
folders = [basefolder / '..' / folder for folder in ('img1', 'img2')]

tiff_stack = Path('examples/data/stack') / 'ImgStack.tif'

# ================== Test contour tracking on ImgSeries ===================

images = ImgSeries(folders, savepath=basefolder)
images.load_times('Img_Files.tsv')  # in case files have changed creation time

ct = ContourTracking(images)
ct.contours.load('Img_ContourTracking')


def test_contour_tracking_basic():
    ct.run()
    assert len(ct.results.data) == 50


def test_contour_tracking_range():
    ct.run(start=10, skip=3)
    assert ct.results.data.shape == (14, 15)


def test_contour_tracking_load():
    ctresults = ContourTrackingResults(savepath=basefolder)
    ctresults.load('Img_ContourTracking')
    assert round(ctresults.data.at[4, 'x3']) == 322


# ================== Test contour tracking on image stack ===================

img_stack = ImgStack(tiff_stack)

ctstack = ContourTracking(img_stack, savepath=basefolder / 'stack')
ctstack.contours.load('Img_ContourTracking')


def test_contourstack_tracking_basic():
    ctstack.run()
    assert ctstack.img_series.data.shape == (200, 100, 112)
    assert len(ctstack.results.data) == 200
