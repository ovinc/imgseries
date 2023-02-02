About
=====

Image analysis tools for image series, based on the following classes:

- `ImgSeries`: general class for image series,
- `GreyLevel`: evolution of average gray level of selected zone(s),
- `ContourTracking`: track objects by contour(s) detection.


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

Below is some information to use the functions available in **imgseries**. Please also consult docstrings and the example Jupyter notebooks *ImgSeries_Static.ipynb*, *ImgSeries_Interactive.ipynb*, *GreyLevel.ipynb* and *ContourTracking.ipynb* for more details and examples.


The management of file series, possibly spread out over multiple folders, follows the scheme of `filo.Series`. In particular, image files are attributed a unique `num` identifier that starts at 0 in the first folder. See `filo` documentation for details. The `ImgSeries` thus directly inherits from `filo.Series`.

*Warning*

If running on a Windows machine and using the parallel option in some of the analysis codes, the call of the function must not be run during import of the python file containing the script (e.g. by using a `if __name__ == '__main__'` block). This is because in Windows, multiprocessing imports the main program when setting up processes, which causes recursive problems. The problem does not arise on Linux / MacOS.


`ImgSeries`: general image series manipulation
----------------------------------------------

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

# Access individual images in the series -------------------------------------
images.files[10]       # filo.File object of image number num=10
images.files[10].file  # actual pathlib.Path file object
images.read(10)        # read image number num=10 into numpy array
images.show(10)        # show image in a matplotlib graph

# Interactive views of image sequence ----------------------------------------
images.animate()       # see image series as a movie (start, end, skip options)
images.inspect()       # browse through image series with a slider (same options)

# Define display options -----------------------------------------------------
# (see details in notebook)
images.contrast.define()
images.colors.define()
images.save_display()  # save rotation and crop parameters in a json file
images.load_display()  # load rotation and crop parameters from json file

# Define global transform applied on all images (rotation + crop) ------------
# (see details in notebook)
images.grayscale.apply = True
images.rotation.define()
images.crop.define()
images.filter.define()
images.subtraction.reference = range(5)  # use 5 first images to subtract to images
images.subtraction.relative = True       # (I - Iref) / I_ref instead of I - Iref
images.active_transforms  # see currently applied transforms on images
images.save_transform()  # save rotation and crop parameters in a json file
images.load_transform()  # load rotation and crop parameters from json file

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

Follow the average grey level (brightness) of one or more selected zones on the image sequence.
The `GreyLevel` class accepts an image sequence (`ImgSeries` type, see above) as an input parameter. See also docstrings and the notebook with examples and details: *GreyLevel.ipynb*

```python
from imgseries import GreyLevel, GreyLevelResults

# Create analysis object -----------------------------------------------------
gl = GreyLevel(images)

# Prepare and run analysis ---------------------------------------------------
gl.zones.define()  # interactively select zones on image
gl.zones.load()    # alternative to define() where zones are loaded from saved metadata

# other alternative to load image series and analysis parameters from saved files
gl.regenerate()

gl.run()    # run actual analysis (parallel computation available);
gl.results  # is an object containing data and metadata of the analysis
gl.save()

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

Follow contours of iso-grey-level on image sequence. The `GreyLevel` class accepts an image sequence (`ImgSeries` type, see above) as an input parameter. See also docstrings and the notebook with examples and details: *ContourTracking.ipynb*

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
ct.save()     # save results (data + metadata) to files (csv, json)

# Interactive views of results -----------------------------------------------
ct.show()      # show result of analysis on a given image (default: first one)
ct.animate()   # see results as a movie (start, end, skip options)
ct.inspect()   # browse through results with a slider (same options)

# Load analysis results afterwards (need save() to have been called) ---------
results = ContourTrackingResults()
results.load()   # load analysis results (data + metadata)
results.data, results.raw_contour_data, results.metadata  # useful attributes
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
- drapo (interactive tools for matplotlib figures) >= 1.2.0


## Python version
- Python >= 3.6 because of f-string formatting

# Author

Olivier Vincent

(olivier.vincent@univ-lyon1.fr)
