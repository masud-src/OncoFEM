# OncoFEM

## Software availability

You can either follow the installation instruction below or use the already pre-installed virtual boxes via the following Links:

- Version 1.0:  https://doi.org/10.18419/darus-3720

## Installation and Machine Requirements

This installation was tested on a virtual box created with a linux mint 21.2 cinnamon, 64 bit system and 8 GB RAM on a local machine (intel cpu i7-9700k with 3.6 GHz, 128 GB RAM). To ensure, the system is ready, it is first updated, upgraded and basic packages are installed via apt.
````bash
sudo apt update
sudo apt upgrade
sudo apt install build-essential python3-pytest gmsh libz-dev cmake libeigen3-dev libgmp-dev libgmp3-dev libmpfr-dev libboost-all-dev python3-pip git
````
- Anaconda needs to be installed. Go to https://anaconda.org/ and follow the installation instructions.
- Run the following command to set up an anaconda environment for oncofem and finally install oncofem on the local system.
````bash
git clone https://github.com/masud-src/OncoFEM/
cd OncoFEM
conda env create -f oncofem.yaml
conda activate oncofem
python3 -m pip install .
````
- Set the global path variable and config file. For Linux and macOS modify run the following lines. In Windows system the script will create a batch file ('set_global_path.bat') in your home directory. Run this file from the Command Prompt. Actualize your system and activate oncofem again. If necessary, change the directories in the config.ini file.
````bash
chmod +x set_config.sh.
./set_config.sh
````
- In order to handle real image data and transform this to readable files, the software package SVMTK package need to be installed by the following code lines or visit https://github.com/SVMTK/SVMTK for comprehensive instructions. 
````bash
cd ..
git clone --recursive https://github.com/SVMTK/SVMTK
cd SVMTK
python3 -m pip install .
cd ..
````

- The tutorial files can be downloaded via
(https://doi.org/10.18419/darus-3679). Please unzip the folder next to the oncofem folder or adjust the relevant
directories in the config.ini file.
- For testing if the installation gone right, go to tutorial and run first or second tutorial
````bash
python3 tut_01_quickstart.py
python3 tut_02_academic_example.py
````
