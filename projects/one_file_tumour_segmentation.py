
from collections import OrderedDict
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import itertools as it
import math
import pprint
import yaml
import os
import numpy as np
from random import randint, random, sample, uniform
import pathlib
from sklearn.model_selection import KFold
import SimpleITK as sitk
import time
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from types import SimpleNamespace
from itertools import combinations, product

HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]
trs = list(combinations(range(2, 5), 2)) + [None]
flips = list(range(2, 5)) + [None]
rots = list(range(1, 4)) + [None]
transform_list = list(product(flips, rots))
NULL_IMAGE = "999_im.nii.gz"

def simple_tta(x):
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function
    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])
    return out

def apply_simple_tta(model, x, average=True):
    todos = simple_tta(x)
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out
    return torch.stack(out).mean(dim=0)

def revert_tta_factory(flip, rot):
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise

def mkdir_if_not_exist(dir: str, exists_ok=True):
    """
    Makes directory if not exists and returns the string

    *Arguments:*
        dir: String

    *Example:*
        dir = mkdir_if_not_exist(dir) 
    """
    from pathlib import Path
    try:
        Path(dir).mkdir(parents=True, exist_ok=exists_ok)
    except (FileExistsError):
        print("Folder already exists")
    return dir

def save_args(args):
    """
    Save parsed arguments to config file. Used for neural net
    """
    config = vars(args).copy()
    del config['save_folder']
    del config['seg_folder']
    pprint.pprint(config)
    config_file = args.save_folder + os.sep + "hyperparam.yaml"
    with open(config_file, "w") as file:
        yaml.dump(config, file)

def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return torch.nn.GroupNorm(groups_, planes)

def get_norm_layer(norm_type="group"):
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None
    else:
        return lambda x: torch.nn.InstanceNorm3d(x, affine=True)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """
    Save Training state.
    """
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)

def reload_ckpt(args, model, optimizer, scheduler):
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

def reload_ckpt_bis(ckpt, model, optimizer=None):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt}' (epoch {start_epoch})")
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")

def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right

def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    if scale != 0.0:
        image = (image - min_) / scale
    return image

def determinist_collate(batch):
    batch = pad_batch_to_max_shape(batch)
    return torch.utils.data._utils.collate.default_collate(batch)

def pad_batch_to_max_shape(batch):
    shapes = (sample['label'].shape for sample in batch)
    _, z_sizes, y_sizes, x_sizes = list(zip(*shapes))
    maxs = [int(max(z_sizes)), int(max(y_sizes)), int(max(x_sizes))]
    for i, max_ in enumerate(maxs):
        max_stride = 16
        if max_ % max_stride != 0:
            # Make it divisible by 16
            maxs[i] = ((max_ // max_stride) + 1) * max_stride
    zmax, ymax, xmax = maxs
    for elem in batch:
        exple = elem['label']
        zpad, ypad, xpad = zmax - exple.shape[1], ymax - exple.shape[2], xmax - exple.shape[3]
        assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
        # free data augmentation
        left_zpad, left_ypad, left_xpad = [random.randint(0, pad) for pad in (zpad, ypad, xpad)]
        right_zpad, right_ypad, right_xpad = [pad - left_pad for pad, left_pad in zip((zpad, ypad, xpad), (left_zpad, left_ypad, left_xpad))]
        pads = (left_xpad, right_xpad, left_ypad, right_ypad, left_zpad, right_zpad)
        elem['image'], elem['label'] = F.pad(elem['image'], pads), F.pad(elem['label'], pads)
    return batch

def pad_batch1_to_compatible_size(batch):
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)

def pad_single_to_compatible_size(batch):
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(1), ymax - batch.size(2), xmax - batch.size(3)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)

def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

def zscore_normalise(img: np.ndarray) -> np.ndarray:
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img

def remove_unwanted_background(image, threshold=1e-5):
    """Use to crop zero_value pixel from MRI image.
    """
    dim = len(image.shape)
    non_zero_idx = np.nonzero(image > threshold)
    min_idx = [np.min(idx) for idx in non_zero_idx]
    # +1 because slicing is like range: not inclusive!!
    max_idx = [np.max(idx) + 1 for idx in non_zero_idx]
    bbox = tuple(slice(_min, _max) for _min, _max in zip(min_idx, max_idx))
    return image[bbox]

def random_crop2d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    if len(set(tuple(image.shape) for image in images)) > 1:
        raise ValueError("Image shapes do not match")
    shape = images[0].shape
    new_sizes = [int(dim * random.uniform(min_perc, max_perc)) for dim in shape]
    min_idx = [random.randint(0, ax_size - size) for ax_size, size in zip(shape, new_sizes)]
    max_idx = [min_id + size for min_id, size in zip(min_idx, new_sizes)]
    bbox = list(slice(min_, max(max_, 1)) for min_, max_ in zip(min_idx, max_idx))
    # DO not crop channel axis...
    bbox[0] = slice(0, shape[0])
    # prevent warning
    bbox = tuple(bbox)
    cropped_images = [image[bbox] for image in images]
    if len(cropped_images) == 1:
        return cropped_images[0]
    else:
        return cropped_images

def random_crop3d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    return random_crop2d(min_perc, max_perc, *images)

def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(np.l_and(np.l_not(preds[i]), np.l_not(targets[i])))
            fp = np.sum(np.l_and(preds[i], np.l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(np.l_and(preds[i], targets[i]))
            tn = np.sum(np.l_and(np.l_not(preds[i]), np.l_not(targets[i])))
            fp = np.sum(np.l_and(preds[i], np.l_not(targets[i])))
            fn = np.sum(np.l_and(np.l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list

def save_metrics(epoch, metrics, swa, writer, current_epoch, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{'_swa' if swa else ''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)

def update_teacher_parameters(model, teacher_model, global_step, alpha=0.99 / 0.999):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for teacher_param, param in zip(teacher_model.parameters(), model.parameters()):
        teacher_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    # print("teacher updated!")

def randomise_blanks(rand, paths):
    """
    t.b.d.
    """
    modified_list = paths.copy()
    if rand:
        num_blanks = randint(0, len(paths) - 1)
        indices = sample(range(len(paths)), num_blanks)
        for index in indices:
            modified_list[index] = NULL_IMAGE
    return modified_list

def get_datasets(folder, patterns, seed, debug, rand_blank=False, no_seg=False, full=False, fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(folder).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    if full:
        train_dataset = Brats(patients_dir, patterns, rand_blank, training=True, debug=debug, normalisation=normalisation)
        bench_dataset = Brats(patients_dir, patterns, rand_blank, training=False, benchmarking=True, debug=debug,
                              normalisation=normalisation)
        return train_dataset, bench_dataset
    if no_seg:
        return Brats(patients_dir, patterns, rand_blank, training=False, debug=debug, no_seg=no_seg, normalisation=normalisation)
    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    print("first idx of train", train_idx[0])
    print("first idx of test", val_idx[0])
    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_idx]
    # return patients_dir
    train_dataset = Brats(train, patterns, rand_blank, training=True,  debug=debug, normalisation=normalisation)
    val_dataset = Brats(val, patterns, rand_blank, training=False, data_aug=False,  debug=debug, normalisation=normalisation)
    bench_dataset = Brats(val, patterns, rand_blank, training=False, benchmarking=True, debug=debug, normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset

def generate_segmentations(data_loader, model, writer, args):
    metrics_list = []
    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if model.deep_supervision:
                    pre_segs, _ = model(inputs)
                else:
                    pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)
        # remove pads
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        segs = torch.zeros((1, 3, 155, 240, 240))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)
        print("Writing " + str(args.seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
        sitk.WriteImage(labelmap, str(args.seg_folder) + str(os.sep) + str(patient_id) + ".nii.gz")
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer.add_scalar("benchmark_" + str(metric) + str(os.sep) + str(label), score)
    df.to_csv(args.save_folder + os.sep + "results.csv", index=False)

class Brats(torch.utils.data.dataset.Dataset):
    def __init__(self, patients_dir, patterns, rand_blank, benchmarking=False, training=True, debug=False, data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.debug = debug
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = patterns
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [str(patient_dir) + os.sep + str(patient_id) + str(value) + ".nii.gz" for value in self.patterns]
            paths = randomise_blanks(rand_blank, paths)
            patient = dict((x.replace("_", ""), paths[i]) for i, x in enumerate(self.patterns))
            patient["id"] = patient_id
            patient["seg"] = str(patient_dir) + os.sep + str(patient_id) + "_seg.nii.gz" if not no_seg else None
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, 
                    label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas) if not self.debug else 3

class ConvBnRelu(torch.nn.Sequential):
    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict([
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', torch.nn.ReLU(inplace=True)),
                        ('dropout', torch.nn.Dropout(p=dropout)),
                            ])
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict([
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', torch.nn.ReLU(inplace=True)),
                        ('dropout', torch.nn.Dropout(p=dropout)),
                    ])
            )

class UBlock(torch.nn.Sequential):
    """
    Unet mainstream downblock.
    """
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            OrderedDict([
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )

class UBlockCbam(torch.nn.Sequential):
    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlockCbam, self).__init__(
            OrderedDict([
                    ('UBlock', UBlock(inplanes, midplanes, outplanes, norm_layer, dilation=dilation, dropout=dropout)),
                    ('CBAM', CBAM(outplanes, norm_layer=norm_layer)),
                ])
        )

class BasicConv(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, norm_layer=None):
        super(BasicConv, self).__init__()
        bias = False
        self.out_channels = out_planes
        self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                                    padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = norm_layer(out_planes)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(gate_channels, gate_channels // reduction_ratio),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class ChannelPool(torch.nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(torch.nn.Module):
    def __init__(self, norm_layer=None):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, norm_layer=norm_layer)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM(torch.nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, norm_layer=None):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate(norm_layer)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class Unet(torch.nn.Module):
    """
    Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = torch.nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = torch.nn.Sequential(conv1x1(features[3], num_classes), torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = torch.nn.Sequential(conv1x1(features[2], num_classes), torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = torch.nn.Sequential(conv1x1(features[1], num_classes), torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = torch.nn.Sequential(conv1x1(features[0], num_classes), torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.InstanceNorm3d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder
        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip([bottom, bottom_2, up3, up2], [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out

class EquiUnet(Unet):
    """
    Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = torch.nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = torch.nn.Sequential(
                conv1x1(features[3], num_classes),
                torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = torch.nn.Sequential(
                conv1x1(features[2], num_classes),
                torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = torch.nn.Sequential(
                conv1x1(features[1], num_classes),
                torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = torch.nn.Sequential(
                conv1x1(features[0], num_classes),
                torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = torch.nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = torch.nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = torch.nn.Sequential(
                conv1x1(features[3], num_classes),
                torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = torch.nn.Sequential(
                conv1x1(features[2], num_classes),
                torch.nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = torch.nn.Sequential(
                conv1x1(features[1], num_classes),
                torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = torch.nn.Sequential(
                conv1x1(features[0], num_classes),
                torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

class WeightSWA(object):
    """
    SWA or fastSWA
    Taken from https://github.com/benathi/fastswa-semi-sup
    """

    def __init__(self, swa_model):
        self.num_params = 0
        self.swa_model = swa_model  # assume that the parameters are to be discarded at the first update

    def update(self, student_model):
        self.num_params += 1
        print("Updating SWA. Current num_params =", self.num_params)
        if self.num_params == 1:
            print("Loading State Dict")
            self.swa_model.load_state_dict(student_model.state_dict())
        else:
            inv = 1.0 / float(self.num_params)
            for swa_p, src_p in zip(self.swa_model.parameters(), student_model.parameters()):
                swa_p.data.add_(-inv * swa_p.data)
                swa_p.data.add_(inv * src_p.data)

    def reset(self):
        self.num_params = 0

class EDiceLoss(torch.nn.Module):
    """
    Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices

class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.9, 0.999), eps=1e-8, weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        for group in self.param_groups:
            group["step_counter"] = 0  # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]

        # don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # ------------ radam
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > 4:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        # ---------------- end radam step

        # look ahead tracking and updating if latest batch = k
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)

        return loss

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class DataAugmenter(torch.nn.Module):
    """
    Performs random flip and rotation batch wise, and reverse it if needed.
    """
    def __init__(self, p=0.5, noise_only=False, channel_shuffling=False, drop_channnel=False):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False
        self.noise_only = noise_only
        self.channel_shuffling = channel_shuffling
        self.drop_channel = drop_channnel

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:
                x = x * uniform(0.9, 1.1)
                std_per_channel = torch.stack(list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1))))
                noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel]).to(x.device)
                x = x + noise
                if random() < 0.2 and self.channel_shuffling:
                    new_channel_order = sample(range(x.size(1)), x.size(1))
                    x = x[:, new_channel_order]
                    print("channel shuffling")
                if random() < 0.2 and self.drop_channel:
                    x[:, sample(range(x.size(1)), 1)] = 0
                    print("channel Dropping")
                if self.noise_only:
                    return x
                self.transpose = sample(range(2, x.dim()), 2)
                self.flip = randint(2, x.dim() - 1)
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            if isinstance(x, list):  # case of deep supervision
                seg, deeps = x
                reversed_seg = seg.flip(self.flip).transpose(*self.transpose)
                reversed_deep = [deep.flip(self.flip).transpose(*self.transpose) for deep in deeps]
                return reversed_seg, reversed_deep
            else:
                return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x


class TrainParam:
    def __init__(self):
        self.data_folder = None
        self.save_folder = None
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.rand_blank = False
        self.input_channel = None
        self.output_channel = 3
        self.arch = "EquiUnet"
        self.width = 2
        self.workers = 2
        self.start_epoch = 0
        self.epochs = 200
        self.batch_size = 1
        self.lr = 0.0001
        self.weight_decay = 0.0
        self.resume = False
        self.debug = False
        self.deep_sup = False
        self.no_fp16 = False
        self.warm = 3
        self.val = 3
        self.fold = 0
        self.norm_layer = "group"
        self.swa = False
        self.swa_repeat = 5
        self.optim = "ranger"
        self.com = None
        self.dropout = 0.0
        self.warm_restart = False
        self.full = False

class InferParam:
    def __init__(self):
        self.config = "/data/default_nn/default_nn.yaml"
        self.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.input_data = None
        self.normalisation = "minmax"
        self.output_path = None
        self.on = "train"
        self.input = None  # former on
        self.tta = False

class TumorSegmentation:

    def __init__(self):
        self.devices = "0"
        self.seed = 16111990
        self.dict_models = {"EquiUnet": EquiUnet}

        self.train_param = TrainParam()
        self.infer_param = InferParam()

    def run_training(self):
        """ 
        The main training function.

        Only works for single node (be it single or multi-GPU)
        """
        # setup
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        print("Working with " + str(ngpus) + " GPUs")
        if self.train_param.optim.lower() == "ranger":
            self.train_param.warm = 0

        mkdir_if_not_exist(self.train_param.save_folder)
        self.train_param.seg_folder = self.train_param.save_folder + os.sep + "segs"
        mkdir_if_not_exist(self.train_param.seg_folder)
        save_args(self.train_param)
        t_writer = SummaryWriter(str(self.train_param.save_folder))

        # Create model
        print("Creating " + str(self.train_param.arch))
        self.train_param.input_channel = len(self.train_param.input_patterns)

        model_maker = self.dict_models[self.train_param.arch]

        model = model_maker(self.train_param.input_channel, self.train_param.output_channel,
                            width=self.train_param.width, deep_supervision=self.train_param.deep_sup,
                            norm_layer=get_norm_layer(self.train_param.norm_layer), dropout=self.train_param.dropout)

        print("total number of trainable parameters " + str(count_parameters(model)))

        if self.train_param.swa:
            # Create the average model
            swa_model = model_maker(self.train_param.input_channel, self.train_param.output_channel,
                                    width=self.train_param.width, deep_supervision=self.train_param.deep_sup,
                                    norm_layer=get_norm_layer(self.train_param.norm_layer))
            for param in swa_model.parameters():
                param.detach_()
            swa_model = swa_model.cuda()
            swa_model_optim = WeightSWA(swa_model)

        if ngpus > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        print(model)
        model_file = self.train_param.save_folder + os.sep + "model.txt"
        with open(model_file, "w") as f:
            print(model, file=f)

        criterion = EDiceLoss().cuda()
        metric = criterion.metric
        print(metric)

        rangered = False  # needed because LR scheduling scheme is different for this optimizer
        if self.train_param.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.train_param.lr, 
                                         weight_decay=self.train_param.weight_decay, eps=1e-4)

        elif self.train_param.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.train_param.lr, 
                                        weight_decay=self.train_param.weight_decay, momentum=0.9, nesterov=True)
        elif self.train_param.optim == "adamw":
            print("weight decay argument will not be used. Default is 11e-2")
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_param.lr)

        elif self.train_param.optim == "ranger":
            optimizer = Ranger(model.parameters(), lr=self.train_param.lr, weight_decay=self.train_param.weight_decay)
            rangered = True

        # optionally resume from a checkpoint
        if self.train_param.resume:
            reload_ckpt(self.train_param, model, optimizer)

        if self.train_param.debug:
            self.train_param.epochs = 2
            self.train_param.warm = 0
            self.train_param.val = 1

        if self.train_param.full:
            train_dataset, bench_dataset = get_datasets(self.train_param.data_folder, self.train_param.input_patterns,
                                                        self.train_param.seed, self.train_param.debug, 
                                                        rand_blank=self.train_param.rand_blank, full=True)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_param.batch_size, 
                                                       shuffle=True, num_workers=self.train_param.workers, 
                                                       pin_memory=False, drop_last=True)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=self.train_param.workers)

        else:
            train_dataset, val_dataset, bench_dataset = get_datasets(self.train_param.data_folder, self.train_param.input_patterns,
                                                                     self.seed, self.train_param.debug,
                                                                     rand_blank=self.train_param.rand_blank, fold_number=self.train_param.fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_param.batch_size, 
                                                       shuffle=True, num_workers=self.train_param.workers, 
                                                       pin_memory=False, drop_last=True)

            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=max(1, self.train_param.batch_size // 2), 
                                                     shuffle=False, pin_memory=False, 
                                                     num_workers=self.train_param.workers, collate_fn=determinist_collate)

            bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=self.train_param.workers)
            print("Val dataset number of batch:", len(val_loader))

        print("Train dataset number of batch:", len(train_loader))
        # create grad scaler
        scaler = torch.cuda.amp.GradScaler()

        # Actual Train loop
        best = np.inf
        print("start warm-up now!")
        if self.train_param.warm != 0:
            tot_iter_train = len(train_loader)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cur_iter: (1 + cur_iter)/(tot_iter_train * self.train_param.warm))

        patients_perf = []

        if not self.train_param.resume:
            for epoch in range(self.train_param.warm):
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, 
                                          epoch, t_writer, scaler, scheduler, save_folder=self.train_param.save_folder,
                                          no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.train_param.val == 0 and not self.train_param.full:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, epoch,
                                                    t_writer, save_folder=self.train_param.save_folder,
                                                    no_fp16=self.train_param.no_fp16)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

        if self.train_param.warm_restart:
            print('Total number of epochs should be divisible by 30, else it will do odd things')
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, eta_min=1e-7)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   self.train_param.epochs + 30 if not rangered else round(
                                                                       self.train_param.epochs * 0.5))
        print("start training now!")
        if self.train_param.swa:
            # c = 15, k=3, repeat = 5
            c, k, repeat = 30, 3, self.train_param.swa_repeat
            epochs_done = self.train_param.epochs
            reboot_lr = 0
            if self.train_param.debug:
                c, k, repeat = 2, 1, 2

        for epoch in range(self.train_param.start_epoch + self.train_param.warm, self.train_param.epochs + self.train_param.warm):
            try:
                # do_epoch for one epoch
                ts = time.perf_counter()
                model.train()
                training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer, epoch, t_writer,
                                     scaler, save_folder=self.train_param.save_folder,
                                     no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                te = time.perf_counter()
                print("Train Epoch done in " + str(te - ts) + " s")

                # Validate at the end of epoch every val step
                if (epoch + 1) % self.train_param.val == 0 and not self.train_param.full:
                    model.eval()
                    with torch.no_grad():
                        validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                               epoch,
                                               t_writer,
                                               save_folder=self.train_param.save_folder,
                                               no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)

                    t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, epoch)

                    if validation_loss < best:
                        best = validation_loss
                        model_dict = model.state_dict()
                        save_checkpoint(
                            dict(
                                epoch=epoch, arch=self.train_param.arch,
                                state_dict=model_dict,
                                optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),
                            ),
                            save_folder=self.train_param.save_folder, )

                    ts = time.perf_counter()
                    print("Val epoch done in " + str(ts - te) + " s")

                if self.train_param.swa:
                    if (self.train_param.epochs - epoch - c) == 0:
                        reboot_lr = optimizer.param_groups[0]['lr']

                if not rangered:
                    scheduler.step()
                    print("scheduler stepped!")
                else:
                    if epoch / self.train_param.epochs > 0.5:
                        scheduler.step()
                        print("scheduler stepped!")

            except KeyboardInterrupt:
                print("Stopping training loop, doing benchmark")
                break

        if self.train_param.swa:
            swa_model_optim.update(model)
            print("SWA Model initialised!")
            for i in range(repeat):
                optimizer = torch.optim.Adam(model.parameters(), self.train_param.lr / 2, weight_decay=self.train_param.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c + 10)
                for swa_epoch in range(c):
                    # do_epoch for one epoch
                    ts = time.perf_counter()
                    model.train()
                    swa_model.train()
                    current_epoch = epochs_done + i * c + swa_epoch
                    training_loss = self.step(train_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                         current_epoch, t_writer,
                                         scaler, no_fp16=self.train_param.no_fp16, patients_perf=patients_perf)
                    te = time.perf_counter()
                    print("Train Epoch done in " + str(te - ts) + " s")

                    t_writer.add_scalar("SummaryLoss/train", training_loss, current_epoch)

                    # update every k epochs and val:
                    print("cycle number: " + str(i), "swa_epoch: " + str(swa_epoch), "total_cycle_to_do " + str(repeat))
                    if (swa_epoch + 1) % k == 0:
                        swa_model_optim.update(model)
                        if not self.train_param.full:
                            model.eval()
                            swa_model.eval()
                            with torch.no_grad():
                                validation_loss = self.step(val_loader, model, criterion, metric, self.train_param.deep_sup, optimizer,
                                                       current_epoch,
                                                       t_writer, save_folder=self.train_param.save_folder, no_fp16=self.train_param.no_fp16)
                                swa_model_loss = self.step(val_loader, swa_model, criterion, metric, self.train_param.deep_sup, optimizer,
                                                      current_epoch,
                                                      t_writer, swa=True, save_folder=self.train_param.save_folder,
                                                      no_fp16=self.train_param.no_fp16)

                            t_writer.add_scalar("SummaryLoss/val", validation_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/swa", swa_model_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit", validation_loss - training_loss, current_epoch)
                            t_writer.add_scalar("SummaryLoss/overfit_swa", swa_model_loss - training_loss, current_epoch)
                    scheduler.step()
            epochs_added = c * repeat
            save_checkpoint(
                dict(
                    epoch=self.train_param.epochs + epochs_added, arch=self.train_param.arch,
                    state_dict=swa_model.state_dict(),
                    optimizer=optimizer.state_dict()
                ),
                save_folder=self.train_param.save_folder, )
        else:
            save_checkpoint(
                dict(
                    epoch=self.train_param.epochs, arch=self.train_param.arch,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict()
                ),
                save_folder=self.train_param.save_folder, )

        try:
            df_individual_perf = pd.DataFrame.from_records(patients_perf)
            print(df_individual_perf)
            df_individual_perf.to_csv(str(self.train_param.save_folder) + os.sep + "patients_indiv_perf.csv")
            reload_ckpt_bis(str(self.train_param.save_folder) + os.sep + "model_best.pth.tar", model)
            generate_segmentations(bench_loader, model, t_writer, self.train_param)
        except KeyboardInterrupt:
            print("Stopping right now!")

    def step(self, data_loader, model, criterion: EDiceLoss, metric, deep_supervision, optimizer, epoch, writer, scaler=None,
             scheduler=None, swa=False, save_folder=None, no_fp16=False, patients_perf=None):
        # Setup
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        mode = "train" if model.training else "val"
        batch_per_epoch = len(data_loader)
        progress = ProgressMeter(batch_per_epoch, [batch_time, data_time, losses], 
                                 prefix=str(mode) + "Epoch: [" + str(epoch) + "]")

        end = time.perf_counter()
        metrics = []
        print(f"fp 16: {not no_fp16}")
        data_aug = DataAugmenter(p=0.8, noise_only=False, channel_shuffling=False, drop_channnel=True).cuda()

        for i, batch in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.perf_counter() - end)

            targets = batch["label"].cuda(non_blocking=True)
            inputs = batch["image"]
            nan_mask = torch.isnan(inputs)
            inputs = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float16), inputs).cuda()
            patient_id = batch["patient_id"]

            with torch.cuda.amp.autocast(enabled=not no_fp16):
                # data augmentation step
                if mode == "train":
                    inputs = data_aug(inputs)
                if deep_supervision:
                    segs, deeps = model(inputs)
                    if mode == "train":  # revert the data aug
                        segs, deeps = data_aug.reverse([segs, deeps])
                    loss_ = torch.stack([criterion(segs, targets)] + [criterion(deep, targets) for deep in deeps])
                    print(f"main loss: {loss_}")
                    loss_ = torch.mean(loss_)
                else:
                    segs = model(inputs)
                    if mode == "train":
                        segs = data_aug.reverse(segs)
                    loss_ = criterion(segs, targets)
                if patients_perf is not None:
                    patients_perf.append(
                        dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
                    )

                writer.add_scalar(f"Loss/{mode}{'_swa' if swa else ''}",
                                  loss_.item(),
                                  global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                if not model.training:
                    metric_ = metric(segs, targets)
                    metrics.extend(metric_)

            # compute gradient and do SGD step
            if model.training:
                scaler.scale(loss_).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
            if scheduler is not None:
                scheduler.step()

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()
            # Display progress
            progress.display(i)

        if not model.training:
            save_metrics(epoch, metrics, swa, writer, epoch, False, save_folder)

        if mode == "train":
            writer.add_scalar(f"SummaryLoss/train", losses.avg, epoch)
        else:
            writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

        return losses.avg

    def run_segmentation(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.devices
        # setup
        random.seed(self.seed)
        ngpus = torch.cuda.device_count()
        if ngpus == 0:
            raise RuntimeWarning("This will not be able to run on CPU only")

        save_folder = pathlib.Path(self.dir)
        save_folder.mkdir(parents=True, exist_ok=True)

        config_file = pathlib.Path(self.infer_param.config).resolve()
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            args = yaml.safe_load(file)
            args = SimpleNamespace(**args, ckpt=ckpt)
            if not hasattr(args, "normalisation"):
                args.normalisation = "minmax"

        # Create model
        model_maker = self.dict_models[args.arch]
        model = model_maker(args.input_channel, 3, width=args.width, deep_supervision=args.deep_sup, 
                            norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

        reload_ckpt_bis(str(args.ckpt), model)

        dataset_minmax = get_datasets(self.infer_param.input_data, self.infer_param.input_patterns, self.seed, False, no_seg=True, normalisation="minmax")
        dataset_zscore = get_datasets(self.infer_param.input_data, self.infer_param.input_patterns, self.seed, False, no_seg=True, normalisation="zscore")
        loader_minmax = torch.utils.data.DataLoader(dataset_minmax, batch_size=1, num_workers=2)
        loader_zscore = torch.utils.data.DataLoader(dataset_zscore, batch_size=1, num_workers=2)

        print("Val dataset number of batch:", len(loader_minmax))
        for i, (batch_minmax, batch_zscore) in enumerate(zip(loader_minmax, loader_zscore)):
            patient_id = batch_minmax["patient_id"][0]
            ref_img_path = batch_minmax["seg_path"][0]
            crops_idx_minmax = batch_minmax["crop_indexes"]
            crops_idx_zscore = batch_zscore["crop_indexes"]
            inputs_minmax = batch_minmax["image"]
            inputs_zscore = batch_zscore["image"]
            inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
            inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
            model_preds = []
            last_norm = None
            if args.normalisation == last_norm:
                pass
            elif args.normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif args.normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    if self.infer_param.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                        model_preds.append(pre_segs)
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)

                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    print("pre_segs size", pre_segs.shape)
                    segs = torch.zeros((1, 3, 155, 240, 240))
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                    print("segs size", segs.shape)

                    model_preds.append(segs)
                model.cpu()  # free for the next one
            pre_segs = torch.stack(model_preds).mean(dim=0)

            segs = pre_segs[0].numpy() > 0.5

            et = segs[0]
            net = np.logical_and(segs[1], np.logical_not(et))
            ed = np.logical_and(segs[2], np.logical_not(segs[1]))
            labelmap = np.zeros(segs[0].shape)
            labelmap[et] = 4
            labelmap[net] = 1
            labelmap[ed] = 2
            labelmap = sitk.GetImageFromArray(labelmap)

            ref_img = sitk.ReadImage(ref_img_path)
            labelmap.CopyInformation(ref_img)
            self.infer_param.output_path = str(self.dir) + str(patient_id) + ".nii.gz"
            print("Writing " + self.infer_param.output_path)
            sitk.WriteImage(labelmap, self.infer_param.output_path)

run_train = False
if run_train:
    tms = TumorSegmentation()
    tms.train_param.save_folder = "full_neural_net"
    tms.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    tms.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    tms.run_training()

run_train2 = False
if run_train2:
    tms = TumorSegmentation()
    tms.train_param.save_folder = "t1_t2_fl_neural_net"
    tms.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    tms.train_param.input_patterns = ["_t1", "_t2", "_flair"]
    tms.run_training()

run_train3 = True
if run_train3:
    tms = TumorSegmentation()
    tms.train_param.save_folder = "full_rand_neural_net"
    tms.train_param.rand_blank = True
    tms.train_param.data_folder = "/home/marlon/Software/OncoFEM/tutorial/data/BraTS"
    tms.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    tms.run_training()
