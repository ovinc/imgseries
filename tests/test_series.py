"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See ExamplesSeries.ipynb
Jupyter notebook for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import ImgSeries, GreyLevel, ContourTracking


# # =============================== Misc. config ===============================


modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'data/for-tests-do-not-modify'
folders = [basefolder / '..' / folder for folder in ('img1', 'img2')]

stackfolder = Path('data/stack')
stack = stackfolder / 'ImgStack.tif'


# ========================= Test getting image time ==========================


images = ImgSeries(folders, savepath=basefolder)
images.load_info('Img_Files_Saved.tsv')  # in case files have changed creation time

def test_img_time():
    """General test of setting and getting image times."""
    n = 33
    assert round(images.info.loc[n, 'time (unix)']) == 1599832463
    assert images.info.loc[n, 'folder'] == 'img2'
    assert images.info.loc[n, 'filename'] == 'img-00643.png'


def test_img_time_update():
    """Test loading external time data."""
    n = 40
    images.load_time('Img_Files_Rounded.tsv')
    assert images.info.loc[n, 'time (unix)'] == 1599832477
    images.load_info('Img_Files_Saved.tsv')  # resset to previous state


# =============== Test avg gray level analysis on image series ===============


gl = GreyLevel(paths=folders, savepath=basefolder)
gl.zones.load('Img_GreyLevel_Saved')

def test_glevel_analysis_basic():
    gl.run()
    assert len(gl.data) == 50


def test_glevel_analysis_range():
    gl.run(start=10, end=655, skip=3)
    assert gl.data.shape == (14, 6)


glr = GreyLevel(savepath=basefolder)

def test_glevel_results_load():
    data = glr.load('Img_GreyLevel_Saved')
    assert round(data.at[4, 'zone 3']) == 89


# =============== Test avg gray level analysis on image stacks ===============


glstack = GreyLevel(stack=stack, savepath=stackfolder)
glstack.zones.load('Img_GreyLevel_Saved')

def test_glevelstack_analysis_basic():
    glstack.run()
    assert glstack.stack.shape == (200, 100, 112)
    assert len(glstack.data) == 200


# ================== Test contour tracking on image series ===================


ct = ContourTracking(folders, savepath=basefolder)
ct.contours.load('Img_ContourTracking_Saved')

def test_contour_tracking_basic():
    ct.run()
    assert len(ct.data) == 50


def test_contour_tracking_range():
    ct.run(start=10, skip=3)
    assert ct.data.shape == (14, 15)


ctr = ContourTracking(savepath=basefolder)

def test_contour_tracking_load():
    data = ctr.load('Img_ContourTracking_Saved')
    assert round(data.at[4, 'x3']) == 321


# ================== Test contour tracking on image stacks ===================


ctstack = ContourTracking(stack=stack, savepath=stackfolder)
ctstack.contours.load('Img_ContourTracking_Saved')

def test_contourstack_tracking_basic():
    ctstack.run()
    assert ctstack.stack.shape == (200, 100, 112)
    assert len(ctstack.data) == 200
