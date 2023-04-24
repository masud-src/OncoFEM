# OncoFEM
OncoFEM is a software tool 

## Installation

```bash
conda create --name oncofem --all
```

```bash
pip install torch==1.11 vtk==9.1 meshio antspy dcm2niix
```

SVMTK 

```bash
python3.9 -m pip install .
```

CAPTK

brainmage


## Structure of OncoFEM

First of all, OncoFEM splits into five higher ordered packages, that again split into sub-packages.
A brief overview is given:

- **helper:** herein, auxillaries for the implementation of initial boundary value 
          problem can be found. All functions can help in implementing a base model of
          OncoFEM. Also the config.ini lies here and holds important settings for
          OncoFEM. In the constant file, all settings from the config.ini are imported
          and set as project-wide constants. In general.py, functionalities for 
          communication with the system are implemented, e.g. the function splitPath 
          splits the path (directory) from the file name. Within the sub-module of io, 
          functionalities for the in- and output are gathered. Herein, commands for the
          writing of output files and analysis of them can be found.
              
- **interfaces:** In this sub-package are interfaces to other software packages are
          implemented for an easy usage of them. Every interface is set up as a class,
          that can be initialized with their respective default setting. All of these 
          settings than can be modified by changing the attributes of that particular 
          entity and the actual functionality can be performed with a "run" command. 
          Already implemented interfaces are:
  - The "brainmage" package for skull stripping with an ANN, that can be trained by the user.
  - The "dcm2niix" package that converts medical images from dcm to nifti format.
  - The "fsl" package for the use of the "fast" segmentation. Additionally, math and statistical commands are implemented.
  - The "greedy" package performs the co-registration of the image modalities.
  - The "nii2mesh" package can be used to convert nifti files into meshes of different file formats (.pial, .stl, .vtk)
    
- **modelling:** This sub-package again splits into three different packages that 
            contain all core functionalities for performing numerical simulations. 
            Therefore, it is split into:
            
  - ***base_model:*** Herein, the basic model of the tumour with its entities and
                      its environment is formulated. A base_model module is implemented 
                      from that all other models shall derive. This gives a simple 
                      structure to the implemented models and ensures that each model has 
                      the same interfaces to other structures. Already implemented 
                      base_models are:
                
    - *glioblastoma*: This model is based on the theory of porous media
                                and splits the tumor into solid bodies of the active part and
                                necrotic core and mobile cancer cells that can move in a 
                                interstitial fluid with advection and diffusion. The tumor is
                                surrounded by an solid matrix (stroma). All solid bodies can 
                                take stresses and an interplay between all phases can be 
                                implemented by a set of reaction terms. These are formulated 
                                in the "bio_chem_models" package.
                    
    - *simple_solid_tumor*: This is a simple two phase model in the 
                                theory of porous media, where an exchange is implemented via 
                                a production term between the fluid and solid phase.
                        
    - *stochastic_model*: This model is inspired by the glioblastoma 
                                model, but has no physical background. It is based on growth 
                                statistics, that are evaluated from investigations into the 
                                data sets of BraTS, UPENN and  IvyGAP. A growth function can 
                                be defined from bio-chem-models and are applied to the solid 
                                active entity. A respective growth of the edema and necrotic 
                                core are related to that and the growth happens over the 
                                surface of the particular entity. The actual growth direction
                                of that tumor is set by a motion map.
                
    Furthermore, the nonlinear solver for solving the weak forms is hold 
                        in the solver module. Herein, all settings can be made that are 
                        related to the numerical solution.
            
  - **bio_chem_models:** The package contains the implemented bio-chemical 
                          set-up. With this set-up the production terms of the regarded problem
                          can be implemented. Therefore, it is possible to have a look at 
                          particular relations or to compare different set-ups. So far, a 
                          simple growth model is implemented, that includes a nutrient 
                          component to take into account for metabolism, necrosis and 
                          proliferation. Again, a hereditary module "bio_chem_models.py"
                          ensures the interfaces to other code structures of OncoFEM.
            
  - **field_map_generator:** The field map generator transforms patient-specific
                          information, i.e. segmentation of tumor entities and white matter 
                          segmentation, into processable fields with respect to the selected base
                          model. Herein, the actual geometry of a brain is converted into mesh.
                          The user has the option to use either pre-defined workflows for the 
                          tumor and white matter mapping or a customized way. Furthermore, this 
                          sub-package also holds the geometry class, that gathers all 
                          geometrical information of a initial boundary value problem and 
                          provides selected academic geometries for an easy use.
        
- **mri:** The sub-package holds modules for the preprocessing of magnetic-resonance 
                images. First, there is the MRI class, that controls the subdivided tasks. Each particular task has its own module and class. Therefore, a generalisation module harmonizes the strongly varying input data into a comparable scope. The tumor segmentation module evaluates the location and composition of a tumor and the white matter segmentation filters information about heterogenities in the brain.
- **struct:** In this sub-package structure giving modules that are not related to other modules are gathered. The hierarchical structure starts from a 
  - *measure*, that consists of a mri measurement with a particular modality and meta data, such as the date or machine type. 
  - Next entity is the *state*, which gathers multiple measurements of a certain time point. Usually this consists of a set of different modalities.
  - A state corresponds to a *subject*, that of course can have multiple states at certain time steps.
  - Multiple subjects are investigated in a *study*
  - Detached from this hierarchy is the problem module, that holds information about a problem that is solved with OncoFEM.

## General operating instructions

Import of functionalities via import of sub-modules and than access!

##  MRI sub-package

## Modelling sub-package

## Examples

## Citation

