BINoculars
==========

BINoculars is a tool for data reduction and analysis of large sets of surface diffraction data that have been acquired with a 2D X-ray detector. The intensity of each pixel of a 2D-detector is projected onto a 3-dimensional grid in reciprocal lattice coordinates using a binning algorithm. This allows for fast acquisition and processing of high-resolution datasets and results in a significant reduction of the size of the dataset. The subsequent analysis then proceeds in reciprocal space. It has evolved from the specific needs of the ID03 beamline at the ESRF, but it has a modular design and can be easily adjusted and extended to work with data from other beamlines or from other measurement techniques.

This work has been [published](http://dx.doi.org/10.1107/S1600576715009607) with open access in the Journal of Applied Crystallography Volume 48, Part 4 (August 2015)

## Installation

Grab the [latest sourcecode as zip](https://github.com/id03/binoculars/archive/master.zip) or clone the Git repository. Run `binoculars`, `binoculars-fitaid`, `binoculars-gui` or `binoculars-processgui` directly from the command line.


## Usage

The [BINoculars wiki](https://github.com/id03/binoculars/wiki) contains a detailed tutorial to get started.


## Scripting

If you want more complex operations than offered by the command line or GUI tools, you can manipulate BINoculars data directly from Python. Some examples with detailed comments can be found in the [repository](https://github.com/id03/binoculars/tree/master/examples/scripts). The API documentation on the `binoculars` and `binoculars.space` modules can be accessed via pydoc, e.g. run `pydoc -w binoculars binoculars.space` to generate HTML files. 


## Extending BINoculars

If you want to use BINoculars with your beamline, you need to write a `backend` module. The code contains an [example implementation](https://github.com/id03/binoculars/blob/master/BINoculars/backends/example.py) with many hints and comments.
