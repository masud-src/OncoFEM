"""

"""
import oncofem as of


study = of.helper.structure.Study("tut_04")

mri = of.mri.MRI()
mri.work_dir = study.der_dir
mri.set_tumor_segmentation()

run_train = False
if run_train:
    mri.tumor_segmentation.save_model_folder = mri.tumor_segmentation.ts_dir + "full_neural_net"
    mri.tumor_segmentation.model_param.training_data = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    mri.tumor_segmentation.model_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    mri.tumor_segmentation.model_param.width = 1
    mri.tumor_segmentation.model_param.epochs = 1
    mri.tumor_segmentation.run_training()

run_seg = True
if run_seg:
    mri.tumor_segmentation.input_data = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS/"
    mri.tumor_segmentation.init_inference()
    mri.tumor_segmentation.config = mri.tumor_segmentation.ts_dir + "/full_neural_net/hyperparam.yaml"
    mri.tumor_segmentation.run_inference()
