"""
MRI tumor segmentation training tutorial

To initialize the training of the tumor segmentation just a study is set, in order to generate a workspace. Since, the
tumor segmentation is part of the mri module, such an object needs to be initialized and the directory is manually set.

The tumor segmentation module is initialized via a setter method. After initialization of that object the save folder
is set.  The here shown parameters are the default described in T. Henry et al. (https://arxiv.org/abs/2011.01045), and 
can be changed with following commands:

.model_param.arch = "EquiUnet"                                  - Architecture of model ("EquiUnet" only option so far)
.model_param.training_data = "/PATH/TO/DATA/"                   - Directory of training data (must be in BraTS style)
.model_param.full_training_data = False                         - Trains to full training data set
.model_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"] - Select the wanted input channels that should be used.
.model_param.random_blank_image = False                         - Adaptive training, sets randomly training channels blank 
.model_param.output_channel = 3                                 - Chosen number of output channels
.model_param.width = 48                                         - Width of model on first layer, then doubled
.model_param.optimizer = "ranger"                               - Chosen optimizer ("ranger", "adam", "sgd", "adamw")
.model_param.dropout = 0.0                                      - Rate of randomly zeroes of input samples
.model_param.warm_restart = False                               - Switch for pretraining before restart
.model_param.epochs = 200                                       - Number of epochs
.model_param.batch_size = 1                                     - Batch size (leave to 1)
.model_param.lr = 0.0001                                        - Learning rate 
.model_param.weight_decay = 0.0                                 - Decay of the weights
.model_param.no_float_precision_16 = False                      - If True, decreases the float precision
.model_param.norm_layer = "group"                               - Norm layer of a model, only "group" is defined
.model_param.n_warm_epochs = 3                                  - Number of warm up epochs
.model_param.val_step_interval = 3                              - Interval of validation
.model_param.deep_sup = False                                   - Deep supervision 
.model_param.fold = 0                                           - Number of instances contained in your dataset
.model_param.stochastic_weight_averaging = False                - Weight averaging of multiple models
.model_param.stochastic_weight_averaging_repeat = 5             - Repetition interval of weight averaging
.model_param.workers = 2                                        - Number of processors
.model_param.resume = False                                     - Resuming the training of a model (not tested)

The model width is set to 1 for the generation of a cheap model for testing. With a powerful gpu this parameter can be 
increased. The final command starts the training. Note: The code is written by Henry et al. and customized to the here
wanted needs. Therefore, it is written for an intel chip and gpu in combination with cuda, which will result in errors 
when running with amd hardware.
"""
import oncofem as of
########################################################################################################################
# INPUT
study = of.helper.structure.Study("tut_04")
########################################################################################################################
# TUMOR SEGMENTATION
mri = of.mri.MRI()
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()
mri.tumor_segmentation.save_model_folder = mri.tumor_segmentation.ts_dir + "full_neural_net"
mri.tumor_segmentation.model_param.training_data = of.ONCOFEM_DIR + "/data/tutorial/BraTS/"
mri.tumor_segmentation.model_param.width = 1
mri.tumor_segmentation.run_training()
