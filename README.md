# Phenome Force Demo: Find UFOs

A short workflow to demonstrate how to integrate code with PlantIT. Applies a threshold, dilates/closes, and finds contours of (U)nidentified (F)ruit (O)bjects in an image.

## Usage

`docker run -it -v $(pwd):/opt/spg -w /opt/spg computationalplantscience/spg python3 find.py ufos <input file>`

A file is provided for testing: `data/tomatoes1.jpg
