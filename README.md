About
=====

Image inspection and analysis tools for image series, based on the following classes.

*Representation of image sequences:*
- `ImgSeries` (sequence stored in multiple files),
- `ImgStack` (sequence stored in a stack, e.g. tiff or HDF5).
Objects from these classes can also be generated with the `series()` and `stack()` methods, respectively;

*Analysis of image sequences:*
- `GreyLevel`: evolution of average gray level of selected zone(s),
- `ContourTracking`: track objects by contour(s) detection,
- `Front1D`: measure fronts propagating in one direction,
- `Flicker`: analyze image flicker with reference zone(s).
These classes act on `ImgSeries` or `ImgStack` objects.

The package is customizable and designed to easily incorporate modifications and additional, user-defined plugins (e.g. different type of analysis methods); see *Customize_Analysis.ipynb* and *Customize_Images.ipynb* for examples.


Install
=======

#### Method 1
In a terminal:
```bash
pip install git+https://cameleon.univ-lyon1.fr/ovincent/imgseries
```

#### Method 2
Clone or copy the project into a folder, then cd into the folder where *setup.py* is located, and in a terminal:
```bash
pip install .
```

Quick start
===========

Below is some information to use the functions available in **imgseries**. Please also consult docstrings and the example Jupyter notebooks *ImgSeries_Static.ipynb*, *ImgSeries_Interactive.ipynb*, *Analysis_GreyLevel.ipynb* and *Analysis_ContourTracking.ipynb* for more details and examples.


The management of file series, possibly spread out over multiple folders, follows the scheme of `filo.Series`. In particular, image files are attributed a unique `num` identifier that starts at 0 in the first folder. See `filo` documentation for details. The `ImgSeries` thus directly inherits from `filo.Series`.

*Warning*

If running on a Windows machine and using the parallel option in some of the analysis codes, the call of the function must not be run during import of the python file containing the script (e.g. by using a `if __name__ == '__main__'` block). This is because in Windows, multiprocessing imports the main program when setting up processes, which causes recursive problems. The problem does not arise on Linux / MacOS.


`ImgSeries`, `ImgStack`: general image series manipulation
----------------------------------------------------------

See also the notebook with examples and details: *ImgSeries.ipynb*

```python
from imgseries import ImgSeries, series

# ----------------------------------------------------------------------------
# ======= WORKING WITH IMAGE SERIES (distinct, individual image files) =======
# ----------------------------------------------------------------------------

# EITHER:
images = ImgSeries(paths=['img1', 'img2'])  # implicitly, savepath is current directory
# OR:
images = series(paths=['img1', 'img2'])
# (the series() function is a bit more powerful than the ImgSeries class, as it
# allows the user to select caching options for speed improvements, see below)

# Image dimensions in x and y
images.nx, images.ny

# Access individual images in the series -------------------------------------

images.files[10]       # filo.File object of image number num=10
images.files[10].file  # actual pathlib.Path file object
images.read(10)        # read image number num=10 into numpy array
images.show(10)        # show image in a matplotlib graph

# Interactive views of image sequence ----------------------------------------

images.animate()       # see image series as a movie (start, end, skip options)
images.inspect()       # browse through image series with a slider (same options)
images.profile()         # object that also has inspect(), animate() methods etc.

# Define display options (only for grayscale imagee --------------------------
# (see details in notebook)
images.display.define('contrast')  # set vmin, vmax in imshow interactively
images.display.define('colormap')  # set cmap in imshow() interactively

# Manually: (equivalent: images.display.limits = 0, 255)
images.display.vmin, images.display.vmax = 0, 255
images.display.cmap = 'viridis'

images.save_display()  # save rotation and crop parameters in a json file
images.load_display()  # load rotation and crop parameters from json file

# Define global transform applied on all images (rotation + crop) ------------
# (see details in notebook)

# Note: the transforms to be considered and the order with which they are
# applied on the images can be modified by passing the argument
# transforms= in ImgSeries. For example:
# images = series(transforms=('rotation', 'crop', 'filter'))

# Interactive:
images.rotation.define()
images.crop.define()
images.filter.define()
images.threshold.define()

# Manual:
images.grayscale.apply = True
images.rotation.angle = -70
images.crop.zone = (2, 25, 400, 600)
images.filter.size = 10
images.subtraction.reference = range(5)  # avg first 5 images for subtraction
images.subtraction.relative = True       # (I - Iref) / I_ref instead of I - Iref
images.threshold.vmin = 220  # everything below vmin is False, rest True
images.threshold.vmax = 240  # everything below vmax is True, rest False
# Note: threshold.vmin can be combined with threshold.vmax for a bandpass

# Other transform parameters / methods:
images.save_transforms()  # save rotation, crop etc. parameters in a json file
images.load_transforms()  # load rotation, crop etc. parameters from json file
images.active_transforms  # see currently applied transforms on images
images.reset_transforms()  # reset all transforms

# Corrections can also be applied to image sequences, e.g. flicker and shaking
# (See also Flicker analysis class, below)
# Contrary to transforms, corrections can be
images.flicker.load()
images.save_corrections()
images.active_corrections  # see currently applied corrections on images
images.reset_corrections()  # reset all corrections

# Exporting images (with transforms and/or corrections)
images.export()  # see Export.ipynb for examples

# Manage image timestamps ----------------------------------------------------
images.info  # see correspondence num / file info + automatically extracted image time
images.save_info()  # save above info in a csv file
images.load_info()  # Load info from previously saved csv data (overwrites images.files)
images.load_time('Time_File.txt')  # Keep images.files but update its time information with data from an external csv file.

# ----------------------------------------------------------------------------
# ===================== WORKING WITH A .TIFF IMAGE STACK =====================
# ----------------------------------------------------------------------------

images = ImgSeries(stack='ImgStack.tif')

# All methods/attributes above available, except those associated with timestamps
```

### Caching images for speed improvement

```python
images = series(paths=['img1', 'img2'], cache=True)
images.inspect()  # inspection should be significantly faster
```
See *ImgSeries_Caching.ipynb* for examples, details and options (cache size etc.).
By default, caching is disabled because it can lead to significant memory usage for large files.


`GreyLevel`: average grey level analysis in image series
--------------------------------------------------------

Follow the average grey level (brightness) of one or more selected zones (default: whole image) on the image sequence.
The `GreyLevel` class accepts an image sequence (`ImgSeries` type, see above) as an input parameter. See also docstrings and the notebook with examples and details: *Analysis_GreyLevel.ipynb*

```python
from imgseries import GreyLevel, GreyLevelResults

# Create analysis object -----------------------------------------------------
gl = GreyLevel(images)

# Prepare and run analysis ---------------------------------------------------
# NOTE: if no zones are defined, full image is taken as a default
gl.zones.define()  # interactively select zones on image
gl.zones.load()    # alternative to define() where zones are loaded from saved metadata

# other alternative to load image series and analysis parameters from saved files
gl.regenerate()

gl.run()    # run actual analysis (parallel computation available);
gl.results  # is an object containing data and metadata of the analysis
gl.results.save()

# Interactive views of results -----------------------------------------------
gl.show()      # show result of analysis on a given image (default: first one)
gl.animate()   # see results as a movie (start, end, skip options)
gl.inspect()   # browse through results with a slider (same options)

# Load analysis results afterwards (need save() to have been called) ---------
results = GreyLevelResults()
results.load()   # load analysis results (data + metadata)
results.data, results.metadata  # useful attributes
```


`ContourTracking`: object tracking using contours in image series
-----------------------------------------------------------------

Follow contours of iso-grey-level on image sequence. The `GreyLevel` class accepts an image sequence (`ImgSeries` type, see above) as an input parameter. See also docstrings and the notebook with examples and details: *Analysis_ContourTracking.ipynb*

```python
from imgseries import ContourTracking, ContourTrackingResults

# Create analysis object
ct = ContourTracking(images)

# Prepare and run analysis ---------------------------------------------------
ct.threshold.define()  # interactively select threshold level
ct.contours.define()   # interactively select contours at the above level
ct.contours.load()     # alternative to define() where contours are loaded from saved metadata

# other alternative to load image series and analysis parameters from saved files
ct.regenerate()

ct.run()      # run actual analysis (no parallel computation available)
ct.results    # is an object containing data and metadata of the analysis
ct.results.save()     # save results (data + metadata) to files (csv, json)

# Interactive views of results -----------------------------------------------
ct.show()      # show result of analysis on a given image (default: first one)
ct.animate()   # see results as a movie (start, end, skip options)
ct.inspect()   # browse through results with a slider (same options)

# Load analysis results afterwards (need save() to have been called) ---------
results = ContourTrackingResults()
results.load()   # load analysis results (data + metadata)
results.data, results.raw_contour_data, results.metadata  # useful attributes
```

`Front1D`: Analyze 1D propagating fronts with grey level analysis
-----------------------------------------------------------------

Analyze fronts propagating in one direction (e.g., *x*), by averaging grey levels in the other direction (*y*). The program returns a line of pixel values along *x* as a function of time (i.e., a reslice of the data), where each pixel is an average of all other pixels along *y*. The operation and methods are similar to `GreyLevel` or ``ContourTracking` (see above). See also docstrings and the notebook with examples and details: *Analysis_Front1D.ipynb*

```python
from imgseries import Front1D, Front1DResults
f1d = Front1D(images)
f1d.run()
```


`Flicker`: Get image flicker from grey level variations on a zone
-----------------------------------------------------------------

Analyze flicker on images based on the average gray level variation in a reference zone in the image. The operation and methods are very similar to `GreyLevel`, including reference zone definition (see above). See also docstrings and the notebook with examples and details: *Analysis_Flicker.ipynb*

```python
from imgseries import Flicker, FlickerResults
flick = Flicker(images)
flick.zones.define()
flick.run()
```

The results are stored as a ratio, which is by how much the pixel values in the image have to be divided to remove apparent flicker.

Afterwards, these results can be loaded in the image series to correct flicker automatically (as a *corrections* parameter):
```python
images.flicker.load()
```


# Requirements / dependencies

## Python packages

(installed by pip automatically if necessary)
- skimage (scikit-image)
- matplotlib
- numpy
- importlib-metadata
- tqdm (waitbars)
- filo (file series management) >= 1.1
- gittools (get git commit info) >= 0.5
- imgbasics (basic image processing) >= 0.3.0
- drapo (interactive tools for matplotlib figures) >= 1.2.1


## Python version
- Python >= 3.6 because of f-string formatting

# Author

Olivier Vincent

(ovinc.py@mgmail.com)

# License

This software is under the CeCILL-2.1 license, equivalent to GNU-GPL (see https://cecill.info/)

Copyright Olivier Vincent (2021-2024)
(ovinc.py@gmail.com)

This software is a computer program whose purpose is to provide tools for
inspection and analysis of image sequences
(either as individual files or as stacks).

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software. You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author, the holder of the
economic rights, and the successive licensors have only limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading, using, modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean that it is complicated to manipulate, and that also
therefore means that it is reserved for developers and experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and, more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
