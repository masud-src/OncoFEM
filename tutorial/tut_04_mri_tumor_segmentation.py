"""

"""
import oncofem as of


study = of.helper.structure.Study("tut_04")

mri = of.mri.MRI()
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()

mri.tumor_segmentation.save_model_folder = mri.tumor_segmentation.ts_dir + "full_neural_net"
mri.tumor_segmentation.model_param.arch = "EquiUnet"
mri.tumor_segmentation.model_param.training_data = "/media/data/MRI_data/BraTS2020/TrainingData"
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
