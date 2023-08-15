"""

"""
import oncofem as of


study = of.helper.structure.Study("tut_04")

mri = of.mri.MRI()
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()

mri.tumor_segmentation.save_model_folder = mri.tumor_segmentation.ts_dir + "full_neural_net"
mri.tumor_segmentation.model_param.training_data = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
mri.tumor_segmentation.model_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
mri.tumor_segmentation.model_param.width = 1
mri.tumor_segmentation.model_param.epochs = 1
mri.tumor_segmentation.run_training()
