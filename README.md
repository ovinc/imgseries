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

Below is some information to use the functions available in **imgseries**. Please also consult docstrings and the example Jupyter notebooks *Examples_ImgSeries.ipynb*, *Examples_GreyLevel.ipynb* and *Examples_ContourTracking.ipynb* for more details and examples.


The management of file series, possibly spread out over multiple folders, follows the scheme of `filo.Series`. In particular, image files are attributed a unique `num` identifier that starts at 0 in the first folder. See `filo` documentation for details. All classes below are children of `filo.Series`:

- `ImgSeries` (directly inherits from `filo.Series`)
- `GreyLevel` (inherits from `ImgSeries`)
- `ContourTracking` (inherits from `ImgSeries`)

*Warning*

If running on a Windows machine and using the parallel option in some of the analysis codes, the call of the function must not be run during import of the python file containing the script (e.g. by using a `if __name__ == '__main__'` block). This is because in Windows, multiprocessing imports the main program when setting up processes, which causes recursive problems. The problem does not arise on Linux / MacOS.


`ImgSeries`: general image series manipulation
----------------------------------------------

See also the notebook with examples and details: *Examples_ImgSeries.ipynb*

```python
from imgseries import ImgSeries

# ----------------------------------------------------------------------------
# ======= WORKING WITH IMAGE SERIES (distinct, individual image files) =======
# ----------------------------------------------------------------------------

images = ImgSeries(paths=['img1', 'img2'])  # implicitly, savepath is current directory

# Access individual images in the series -------------------------------------
images.files[10]       # filo.File object of image number num=10
images.files[10].file  # actual file object (pathlib.Path)
images.read(10)        # read image number num=10 into numpy array
images.show(10)        # show image in a matplotlib graph

# Define global transform applied on all images (rotation + crop) ------------
images.rotation.define()
images.crop.define()   # (see details in notebook)

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


`GreyLevel`: average grey level analysis in image series
--------------------------------------------------------

See also the notebook with examples and details: *Examples_GreyLevel.ipynb*

```python
from imgseries import GreyLevel

# Run analysis ---------------------------------------------------------------

# If working with image series:
gl = GreyLevel(paths=['img1', 'img2'], savepath='analysis')

# If working with a tiff stack:
gl = GreyLevel(stack='ImgStack.tif', savepath='analysis')

# Define global transform if necessary (see above in ImgSeries)
gl.rotation.define()
gl.crop.define()

# Prepare and run analysis
gl.zones.define()  # interactively select zones on image
gl.zones.load()    # alternative to define() where zones are loaded from saved metadata
gl.run()   # run actual analysis (parallel computation available)
gl.data    # pandas dataframe containing the results --> plot() etc. methods available
gl.save()  # save results to csv file

# Load analysis results afterwards (need save() to have been called) ---------
gl = GreyLevel(savepath='analysis')  # No need to specify paths here
gl.load()        # load analysis results (including time) as pandas DataFrame
gl.zones.load()  # load info (dict) of location of zones analyzed on images
gl.zones.data    # accessible after zones.load() has been called
gl.zones.show()  # show analysis zones on image (numbering starts at 0)

# NOTE: method below only available for image series, not for stacks
gl.load_info()   # if necessary to look back at num / file correspondence
```

See doctrings and Jupyter Notebooks for examples and method options.


`ContourTracking`: object tracking using contours in image series
-----------------------------------------------------------------

See also the notebook with examples and details: *Examples_ContourTracking.ipynb*

```python
from imgseries import ContourTracking

# Run analysis ---------------------------------------------------------------

ct = ContourTracking(paths=['img1', 'img2'], savepath='analysis')

# If working with a tiff stack:
ct = ContourTracking(stack='ImgStack.tif', savepath='analysis')

# Define global transform if necessary (see above in ImgSeries)
ct.rotation.define()
ct.crop.define()

# Prepare and run analysis
ct.contours.define()  # interactively select contours to follow on image
ct.contours.load()    # alternative to define() where contours are loaded from saved metadata
ct.run()   # run actual analysis (no parallel computation available)
ct.data    # pandas dataframe containing the results --> plot() etc. methods available
ct.save()  # save results to csv file

# Load analysis results afterwards (need save() to have been called) ---------
ct = ContourTracking(savepath='analysis')  # No need to specify paths here
ct.load()        # load analysis results (including time) as pandas DataFrame
ct.contours.load()  # load info (dict) of contour tracking parameters
ct.contours.data    # accessible after contours.load() has been called
ct.contours.show()  # Show contours on reference image used to select them

# NOTE: method below only available for image series, not for stacks
ct.load_info()   # if necessary to look back at num / file correspondence
```

See doctrings and Jupyter Notebooks for examples and method options.


# Requirements / dependencies

## Python packages

(installed by pip automatically if necessary)
- skimage (scikit-image)
- matplotlib
- numpy
- importlib-metadata
- tqdm (waitbars)
- imgbasics (basic image processing)
- filo (file series management)
- gittools (get git commit info)
- drapo (interactive tools for matplotlib figures)


## Python version
- Python >= 3.6 because of f-string formatting

# Author

Olivier Vincent

(olivier.vincent@univ-lyon1.fr)
