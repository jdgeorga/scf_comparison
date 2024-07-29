import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="moirecompare",
    version="1.0.0",
    author="Johnathan Georgaras",
    author_email="jdgeorga@stanford.edu",
    description="Comparing moire structures from QE/LAMMPS/ALLEGRO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
        "nequip>=0.6.1",
        "scipy>=1.13"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)