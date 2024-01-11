"""
YourPackageName - A Python package for atomic structure analysis.

This package provides a set of tools for reading, analyzing, and visualizing atomic structures, 
especially useful in the field of computational materials science.
"""

# Import submodules here if you want them to be accessible directly from the package level
from .io import file_operations
from .analysis import force_analysis, prdf_analysis, displacement_analysis
from .visualization import plotting
from .utils import utilities

# You can also define some package-level constants or functions here if needed

