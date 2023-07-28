# OncoFEM

OncoFEM is a software tool to perform numerical simulations of tumours based on medical image data, providing a possible 
tumour evolution. The software is written to speed up the development towards an increasing demand for patient-specific 
simulations, with the ultimate goal of supporting clinicians in their treatment planning, i. e. medication, surgical 
interventions, classifying of severness. The structure and workflow of OncoFEM is kept general, to be open for the 
inclusion of different types of tumours, organs or tissues. Nevertheless its initial implementation is written for the 
simulation of diffusive astrocytomas (brain tumour), such as Glioblastoma multiforme (GBM). The software divides into 
the preprocessing of medical images and a simulation core module.

![alt text](workflow.png)

Already implemented are a pre-processing entity, that homogenises MRI input data and segments the tumour and 
heterogeneous compartments of the brain. Numerical calculations can be performed by a combination of a macroscopic base 
model with process models on the microscale, that mimic the cell behaviour of cells and cell cohorts. For demonstration 
already the implementation of a two-phase model in the continuum-mechanical framework of the Theory of Porous Media is 
chosen, according to the tumour microenvironment based on Wolf et al. (doi: 10.1117/12.535112). Herein, the problem is 
modelled with a porous approach of a solid extracellular matrix and an intercranical fluid, wherein mobile cancer cells 
are resolved and measured with a molar concentration. The processes on the microscale are assumed with a logistic 
Verhulst equation for the mobile cancer cells that can be coupled to a solid growth term. In case of growing tumour mass 
fluid will be accumulated in the affected areas and a swelling can be observed. 

The software provides a tutorial to learn the basic functionalities. More information can be found in the respective 
paper.

## Requirements

- Python 3
- Anaconda packaging manager 
- Git 
- Git LFS

## Installation

sudo apt install build-essential libz-dev git-lfs cmake libeigen3-dev libgmp-dev libmpfr-dev libboost-dev

anaconda download and follow installation instruction on  
cd anaconda3/bin
./conda init

conda env create -n oncofem -f oncofem.txt


Before the installation of OncoFEM can be done, required software needs to be downloaded and installed. Download and 
build the nii2mesh package with
````bash
git clone https://github.com/neurolabusc/nii2mesh
cd nii2mesh/src
make
````
For the installation of CaPTk, follow the installation shown on the respective github (https://github.com/CBICA/CaPTk).
For installation of brain mage execute the following code lines or visit https://github.com/CBICA/BrainMaGe for further 
instructions.
````bash
git clone https://github.com/CBICA/BrainMaGe.git
cd BrainMaGe
git lfs pull
conda env create -f requirements.yml # create a virtual environment named brainmage
conda activate oncofem # activate it
latesttag=$(git describe --tags) # get the latest tag [bash-only]
echo checking out ${latesttag}
git checkout ${latesttag}
python setup.py install # install dependencies and BrainMaGe
````
Lastly, the SVMTK package need to be downloaded and installed. Execute the following code lines or visit  
https://github.com/SVMTK/SVMTK for comprehensive instructions.
````bash
git clone --recursive https://github.com/SVMTK/SVMTK
git pull
git submodule update --init
python3 -m pip install .
````
Change the following directories in the config.ini file. The first two paths set the directory to the nii2mesh and CaPTk
software packages. The third defines the workspace for your studies. If you want to train own neural networks you can 
adjust the last path. Herein, you can find the runs.
````bash
NII2MESH_DIR: /home/marlon/Software/nii2mesh/src/
CAPTK_DIR: /home/marlon/Software/CaPTk/1.8.1/captk
STUDIES_DIR: /media/marlon/data/studies/
TUMOR_SEGMENTATION_TRAINING_RUN: /media/marlon/data/run/
````
Ensure to have an up-to-date version of setuptools with 
````bash
python -m pip install --upgrade setuptools
````

```bash
conda create --name oncofem --all
```

```bash
pip install torch==1.11 vtk==9.1 meshio antspy dcm2niix
```


## How to

### Implement a base model

### Implement a micro model

## About

OncoFEM is written by Marlon Suditsch
