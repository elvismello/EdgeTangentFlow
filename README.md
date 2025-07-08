# Edge Tangent Flow
This is the implementation of edge tangent flow which is written in "Coherent Line Drawing" by Henry Kang et al, Proc. NPAR 2007.<br>
http://umsl.edu/mathcs/about/People/Faculty/HenryKang/coon.pdf

Edge Tangent Flow is the the method for constructing a smooth direction field that preserves the flow of the salient image features.

# Sample
https://raw.github.com/wiki/naotokimura/edge_tangent_flow/images/ezgif.com-crop.gif

# Requirement
* Numpy
* Pillow
* Scipy


# About this fork
Removed opencv dependencies for the use of scientific libraries, such as scipy and numpy for its functions.

Modified code in order to obtain more performance, vectorizing many sections, such as the function computeNewVector. refine_ETF was also modified for paralelization.

Many parts of the code were slow in Python, being written in a format similar to C, with many nested fors. This version should be more Pythonic (maybe), not being slowed as much.

The the global variable refinedETF was no longer needed.
