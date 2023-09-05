"""
MRI tumor segmentation tutorial

To initialize the training of the tumor segmentation just a study is set, in order to generate a workspace. Since, the
tumor segmentation is part of the mri module, such an object needs to be initialized and the directory is manually set.

The tumor segmentation module is initialized via a setter method. After initialization of that object the save folder
is set. In general, all preferences could be filled via a loop over listed elements and in that way several models could
be trained. The here shown parameters are the default described in T. Henry et al. (https://arxiv.org/abs/2011.01045),
apart from the model width. This is set to 1 for the generation of a cheap model for testing. With a powerful gpu this 
parameter can be increased. The final command starts the training. Note: The code is written mostly by Henry et al.,
they worked with a intel chip and gpu in combination with cuda, which will result in errors when running with amd 
hardware.
"""
import oncofem as of
########################################################################################################################
# INPUT
study = of.helper.structure.Study("tut_04")

mri = of.mri.MRI()
mri.work_dir = study.der_dir
########################################################################################################################
# TUMOR SEGMENTATION
mri.set_tumor_segmentation()
mri.tumor_segmentation.save_model_folder = mri.tumor_segmentation.ts_dir + "full_neural_net"
mri.tumor_segmentation.model_param.arch = "EquiUnet"
mri.tumor_segmentation.model_param.training_data = of.ONCOFEM_DIR + "/data/tutorial/BraTS/"
mri.tumor_segmentation.model_param.full_training_data = False
mri.tumor_segmentation.model_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
mri.tumor_segmentation.model_param.random_blank_image = False
mri.tumor_segmentation.model_param.output_channel = 3
mri.tumor_segmentation.model_param.width = 1
mri.tumor_segmentation.model_param.optimizer = "ranger"
mri.tumor_segmentation.model_param.dropout = 0.0
mri.tumor_segmentation.model_param.warm_restart = False
mri.tumor_segmentation.model_param.epochs = 200
mri.tumor_segmentation.model_param.batch_size = 1
mri.tumor_segmentation.model_param.lr = 0.0001
mri.tumor_segmentation.model_param.weight_decay = 0.0
mri.tumor_segmentation.model_param.no_float_precision_16 = False
mri.tumor_segmentation.model_param.norm_layer = "group"
mri.tumor_segmentation.model_param.n_warm_epochs = 3
mri.tumor_segmentation.model_param.val_step_intervall = 3
mri.tumor_segmentation.model_param.deep_sup = False
mri.tumor_segmentation.model_param.fold = 0
mri.tumor_segmentation.model_param.stochastic_weight_averaging = False
mri.tumor_segmentation.model_param.stochastic_weight_averaging_repeat = 5
mri.tumor_segmentation.model_param.workers = 2
mri.tumor_segmentation.model_param.resume = False
mri.tumor_segmentation.run_training()
