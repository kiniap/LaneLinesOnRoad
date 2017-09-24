# LaneLinesOnRoad

LaneLinesOnRoad implements a pipeline in Python3 to identify the lane lines on the road on images or videos from a camera mounted in a car. The pipeline marks the lane line on the input image or video. A writeup.ipynb explains the various steps in the pipleine and the methodology.

## Packages needed for Python 3
* OpenCv
* Numpy
* MatplotLib
* MoviePy

## Running the pipeline

** command line **

`> python3 findLaneLinesOnRoad.py`

By default it loops through all the images in the test_images folder and creates images with the lane lines drawn in the test_images_output folder


** Anaconda and Jupyter notebooks

`jupyter notebook findLaneLinesOnRoad.ipynb`

## Issues

* Slight Jitter in the lane lines when running on videos
