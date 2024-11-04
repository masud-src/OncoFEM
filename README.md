# OncoFEM

## Software availability

You can either follow the installation instruction below or use the already pre-installed virtual boxes via the following Links:

- Version 1.0:  https://doi.org/10.18419/darus-3720

## Installation and Machine Requirements

This installation was tested on a virtual box created with a linux mint 21.2 cinnamon, 64 bit system and 8 GB RAM on a local machine (intel cpu i7-9700k with 3.6 GHz, 128 GB RAM). To ensure, the system is ready, it is first updated, upgraded and basic packages are installed via apt.
````bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential python3-pytest gmsh libz-dev cmake libeigen3-dev libgmp-dev libmpfr-dev libboost-dev python3-pip git
````
- Anaconda needs to be installed. Go to https://anaconda.org/ and follow the installation instructions.
- Run the following command to set up an anaconda environment for oncofem.
````bash
git clone https://github.com/masud-src/OncoFEM/
cd OncoFEM
conda create --name oncofem --file oncofem.txt
conda activate oncofem
````
- Ensure to have an up-to-date version of setuptools and finally install oncofem on the local system
````bash
python3 -m pip install .
````
- SVMTK package is installed by the following code lines or visit https://github.com/SVMTK/SVMTK for comprehensive instructions.
````bash
cd ..
git clone --recursive https://github.com/SVMTK/SVMTK
cd SVMTK
python3 -m pip install .
cd ..
````
- Set the global variables in your bashrc file
by adding the following line.
````bash
export ONCOFEM=PATH/TO/OncoFEM
````
- Actualize your system and activate oncofem again
````bash
source bashrc
conda activate oncofem
````
- Change the following directories in the config.ini file.
````bash
STUDIES_DIR: /home/onco/studies/
````
- The SRI24 atlases, the tumor segmentation weights and the tutorial files can be downloaded via
(https://doi.org/10.18419/darus-3679). Please unzip the folder next to the oncofem folder or adjust the relevant
directories in the config.ini file.
- For testing if the installation gone right, go to tutorial and run first or second tutorial
````bash
python3 tut_01_quickstart.py
python3 tut_02_academic_example.py
````
