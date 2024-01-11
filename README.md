# SCF COMPARISON

A brief description of what this project does and who it's for.

## Description
This project includes a collection of Python functions and scripts for analyzing atomic structures. It is particularly useful in computational materials science, providing tools for calculating properties like extracting total wall times from Quantum ESPRESSO output files, analyzing forces, and computing various statistical measures.

## Features
- Analyze forces, total energy, atomic positions, and atom types.
- Extract total wall times from Quantum ESPRESSO output files.
- Compute and visualize Partial Radial Distribution Functions (PRDF).
- Calculate and plot structural order parameters, RMSD, MAD, and more.

## Installation
Clone the repository and install the required dependencies:

\```
git clone [repository URL]
cd [repository directory]
pip install -r requirements.txt
\```

## Usage
To use the functions provided in this project, import the required modules in your Python script. For example:

\```python
from analysis_module import compute_prdf, plot_prdf
\```

Refer to individual function docstrings for more detailed usage instructions.

## Examples
Check out the `examples` directory for Jupyter notebooks or scripts demonstrating how to use the various functions in this project.

## Requirements
This project requires the following Python libraries:
- numpy
- matplotlib
- scipy
- ase
- scikit-learn

Ensure these are installed in your environment, or use the `requirements.txt` file to install them.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License
[Choose an appropriate license and include it here]

## Contact
Your Name - jdgeorga@stanford.edu

Project Link: https://github.com/jdgeorga/scf_comparison.git
