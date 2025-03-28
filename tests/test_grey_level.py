"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See example
Jupyter notebooks for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import ImgSeries, ImgStack
from imgseries import GreyLevel, GreyLevelResults


# =============================== Misc. config ===============================

modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'examples/data/for-tests-do-not-modify'
folders = [basefolder / '..' / folder for folder in ('img1', 'img2')]

tiff_stack = Path('examples/data/stack') / 'ImgStack.tif'

# =============== Test avg gray level analysis on image series ===============

images = ImgSeries(folders, savepath=basefolder)
images.load_times('Img_Files.tsv')  # in case files have changed creation time

gl = GreyLevel(images)
gl.zones.load('Img_GreyLevel')


def test_glevel_analysis_basic():
    gl.run()
    assert len(gl.results.data) == 50


def test_glevel_analysis_range():
    gl.run(start=10, end=655, skip=3)
    assert gl.results.data.shape == (14, 6)


def test_glevel_results_load():
    glresults = GreyLevelResults(savepath=basefolder)
    glresults.load('Img_GreyLevel')
    assert round(glresults.data.at[4, 'zone 3']) == 86


# =============== Test avg gray level analysis on image stack ================

img_stack = ImgStack(tiff_stack)

glstack = GreyLevel(img_stack, savepath=basefolder / 'stack')
glstack.zones.load('Img_GreyLevel')


def test_glevelstack_analysis_basic():
    glstack.run()
    assert glstack.img_series.data.shape == (200, 100, 112)
    assert len(glstack.results.data) == 200
