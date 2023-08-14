"""

"""
import oncofem as of

study = of.helper.structure.Study("tut_04")

mri = of.mri.MRI()
mri.study_dir = study.dir
mri.set_tumor_segmentation()

run_train = False
if run_train:
    mri.tumor_segmentation.train_param.save_folder = "full_neural_net"
    mri.tumor_segmentation.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    mri.tumor_segmentation.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    mri.tumor_segmentation.train_param.width = 1
    mri.tumor_segmentation.train_param.epochs = 1
    mri.tumor_segmentation.run_training()

run_seg = True
if run_seg:
    mri.tumor_segmentation.infer_param.input_data = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS/BraTS20_Training_001/"
    mri.tumor_segmentation.infer_param.config = "/home/marlon/Software/OncoFEM/tutorial/full_neural_net/hyperparam.yaml"
    mri.tumor_segmentation.init_inference()
    mri.tumor_segmentation.run_inference()
