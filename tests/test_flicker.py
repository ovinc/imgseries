"""Tests for the imgseries module (pytest), image series analysis programs.

Note: This does not test the interactive options. See ExamplesSeries.ipynb
Jupyter notebook for more details and interactive modes.
"""

# Standard library
from pathlib import Path

# Local imports
import imgseries
from imgseries import ImgSeries
from imgseries import Flicker, FlickerResults


# =============================== Misc. config ===============================

modulefolder = Path(imgseries.__file__).parent / '..'
basefolder = modulefolder / 'data/for-tests-do-not-modify'
folder = basefolder / '..' / 'front-flick'

# ================== Test flicker analysis on image series ===================

images = ImgSeries(folder, savepath=basefolder)

flick = Flicker(images)
flick.zones.load('FlickFront_FlickerData')


def test_flicker_analysis_basic():
    flick.run()
    assert len(flick.results.data) == 25


def test_flicker_analysis_range():
    flick.run(start=2, end=20, skip=2)
    assert flick.results.data.shape == (9, 5)


def test_flicker_results_load():
    flick_results = FlickerResults(savepath=basefolder)
    flick_results.load('FlickFront_FlickerData')
    assert round(flick_results.data.at[4, 'ratio'], 3) == 0.964
