import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import json
from monai.metrics import compute_hausdorff_distance, compute_surface_dice  # HD95 & NSD (Surface Dice)

import SimpleITK as sitk
import numpy as np
import torch

def resample(img, new_spacing=(1.5, 1.5, 1.5), interpolator=sitk.sitkNearestNeighbor):
    """
    Resample a SimpleITK image to isotropic spacing.

    Parameters
    ----------
    img : sitk.Image
        Input image.
    new_spacing : tuple of float
        Desired spacing (sx, sy, sz) in mm.
    interpolator : sitk interpolator
        sitk.sitkNearestNeighbor for labels, sitk.sitkLinear for images.

    Returns
    -------
    resampled_img : sitk.Image
        Resampled image with isotropic spacing.
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interpolator)
    return resample.Execute(img)

def _surface_distance_arrays_mm(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> tuple[np.ndarray, np.ndarray]:
    """
    Return symmetric surface distances in mm:
      d_pred_to_ref (for pred surface points) and d_ref_to_pred (for ref surface points).
    spacing_xyz must be (x, y, z) as provided by SimpleITK.
    """
    # Empty handling up front
    if pred.sum() == 0 and ref.sum() == 0:
        return np.array([0.0]), np.array([0.0])
    if pred.sum() == 0 or ref.sum() == 0:
        return np.array([np.inf]), np.array([np.inf])

    # Convert to SITK with correct spacing
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
    ref_sitk  = sitk.GetImageFromArray(ref.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_xyz)
    ref_sitk.SetSpacing(spacing_xyz)

    # Extract 3D surfaces (fully-connected)
    contour = sitk.LabelContourImageFilter()
    contour.SetFullyConnected(True)
    pred_surf = sitk.GetArrayFromImage(contour.Execute(pred_sitk)) > 0
    ref_surf  = sitk.GetArrayFromImage(contour.Execute(ref_sitk)) > 0

    # Fallback: tiny objects can sometimes produce empty contour; treat any foreground voxel as surface
    if not pred_surf.any() and pred.any():
        pred_surf = pred.astype(bool)
    if not ref_surf.any() and ref.any():
        ref_surf = ref.astype(bool)

    # Distance transforms in mm (to the *boundary*; signed but we take abs)
    dt_ref  = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(ref_sitk,  insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    dt_pred = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(pred_sitk, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))

    d_pred_to_ref = np.abs(dt_ref[pred_surf])
    d_ref_to_pred = np.abs(dt_pred[ref_surf])

    # If either side somehow ends empty, treat as infinite disagreement
    if d_pred_to_ref.size == 0 or d_ref_to_pred.size == 0:
        return np.array([np.inf]), np.array([np.inf])

    return d_pred_to_ref, d_ref_to_pred


def compute_dice(pred, ref):
    """Compute the Dice coefficient for two binary numpy arrays."""
    intersection = np.logical_and(pred, ref).sum()
    denom = pred.sum() + ref.sum()
    if denom == 0:
        # Both empty => define Dice=1.0, or handle as you prefer
        return 1.0
    return 2.0 * intersection / denom

def compute_volume_difference(pred, ref, spacing):
    """
    Compute the volume difference (in mL, for instance) between two binary arrays,
    given spacing = (sx, sy, sz).
    """
    voxel_volume = np.prod(spacing)  # mm^3 per voxel
    vol_pred = pred.sum() * voxel_volume
    vol_ref  = ref.sum()  * voxel_volume
    # convert mm^3 to mL (1 mL = 1000 mm^3)
    return (vol_pred - vol_ref) / 1000.0

def compute_centroid_distance(pred, ref, spacing):
    """
    Compute the Euclidean distance between centroids (in physical space).
    """
    coords_pred = np.argwhere(pred)
    coords_ref  = np.argwhere(ref)
    
    if coords_pred.size == 0 and coords_ref.size == 0:
        # Both empty => distance=0 or handle as you prefer
        return np.nan
    if coords_pred.size == 0 or coords_ref.size == 0:
        # One is empty and the other not => define distance=NaN (or large sentinel)
        return np.nan
    
    centroid_pred = coords_pred.mean(axis=0)  # (z, y, x)
    centroid_ref  = coords_ref.mean(axis=0)   # (z, y, x)
    
    # Convert index space to physical space
    # Make sure to align the order of spacing with the order of the coordinates
    diff = (centroid_pred - centroid_ref) * np.array(spacing[::-1])
    return np.sqrt(np.sum(diff**2))

def compute_hd95(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> float:
    d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
    all_d = np.concatenate([d1, d2])
    # If both arrays are inf (one empty, one non-empty), define as inf (or large sentinel)
    if np.isinf(all_d).all():
        return np.nan
    return float(np.percentile(all_d, 95))

def compute_average_surface_distance(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> float:
    d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
    all_d = np.concatenate([d1, d2])
    if np.isinf(all_d).all():
        return np.nan
    return float(0.5 * (d1.mean() + d2.mean()))

def compute_nsd(pred: np.ndarray, ref: np.ndarray, spacing_xyz, tau_mm: float) -> float:
    """
    Normalized Surface Distance (aka Surface Dice) at tolerance tau_mm.
    Returns a value in [0, 1].

    NSD = ( |S_pred within τ of Ref| + |S_ref within τ of Pred| ) / ( |S_pred| + |S_ref| )
    """
    # Handle empties explicitly (common convention; be explicit in your paper/report)
    if pred.sum() == 0 and ref.sum() == 0:
        return 1.0  # both empty => perfect agreement by convention
    if pred.sum() == 0 or ref.sum() == 0:
        return 0.0  # one empty => total disagreement

    d_pred_to_ref, d_ref_to_pred = _surface_distance_arrays_mm(pred, ref, spacing_xyz)

    # If something went wrong and we only have infs, treat as 0 overlap
    if np.isinf(d_pred_to_ref).all() and np.isinf(d_ref_to_pred).all():
        return 0.0

    within_pred = np.sum(d_pred_to_ref <= tau_mm)
    within_ref  = np.sum(d_ref_to_pred <= tau_mm)
    total_pred  = d_pred_to_ref.size
    total_ref   = d_ref_to_pred.size

    return float((within_pred + within_ref) / (total_pred + total_ref))

def compute_segmentation_metrics(
    preds_dir,
    labels_dir,
    class_list,
    file_suffix=".nii.gz",
    device = torch.device(type='cuda')
):
    """
    1. Finds all segmentation files in `preds_dir` that match names in `labels_dir`.
    2. Computes multiple metrics per class (1..num_classes).
    3. Returns a DataFrame with per-scan metrics and a summary dict with means & stds.
    """
    # List files in each directory
    pred_files = sorted([f for f in os.listdir(preds_dir) if f.endswith(file_suffix)])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(file_suffix)])
    
    # Intersect the two file lists
    common_files = sorted(list(set(pred_files).intersection(set(label_files))))
    print(len(label_files), len(common_files))
    all_metrics = []
    
    for fname in common_files:
        pred_path = os.path.join(preds_dir, fname)
        label_path = os.path.join(labels_dir, fname)
        
        # Load volumes
        pred_img = sitk.ReadImage(pred_path)
        label_img = sitk.ReadImage(label_path)
        # Resample to 1.5 mm isotropic
        # pred_img = resample(pred_img, (1.5, 1.5, 1.5), interpolator=sitk.sitkNearestNeighbor)
        # label_img = resample(label_img, (1.5, 1.5, 1.5), interpolator=sitk.sitkNearestNeighbor)
        pred_array = sitk.GetArrayFromImage(pred_img)   # [z, y, x]
        label_array = sitk.GetArrayFromImage(label_img) # [z, y, x]
        spacing = label_img.GetSpacing()                # (sx, sy, sz)
        # Optional: check shape
        if pred_array.shape != label_array.shape:
            raise ValueError(f"Shape mismatch between prediction and label for {fname}.")

        metrics_per_scan = {"filename": fname}
        
        # Loop over each class 1..num_classes
        num_classes = len(class_list)
        for c in range(1, num_classes+1):
            label_c = (label_array == c)
            pred_c = (pred_array == c)
            
            # If the label is completely missing for this class, set metrics to NaN
            # so that it is ignored in the final average.
            if label_c.sum() == 0:
                dice_c = np.nan
                vol_diff_c = np.nan
                hd95_c = np.nan
                avg_surf_dist_c = np.nan
                centroid_dist_c = np.nan
                nsd = np.nan
            else:
                # Compute metrics normally
                dice_c = compute_dice(pred_c, label_c)
                pred_c_oh = torch.as_tensor(pred_c, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)
                label_c_oh = torch.as_tensor(label_c, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)
                hd95_c = compute_hausdorff_distance(
                    y_pred=pred_c_oh, y=label_c_oh,
                    include_background=True, percentile=95, spacing=(spacing[2], spacing[1], spacing[0])
                ).cpu().numpy()[0][0] 
                nsd = compute_surface_dice(
                    y_pred=pred_c_oh, y=label_c_oh,
                    class_thresholds=[1.0] * 1,
                    include_background=True, spacing=(spacing[2], spacing[1], spacing[0])
                ).cpu().numpy()[0][0] 

            metrics_per_scan[f"dice_class{c}"] = dice_c
            metrics_per_scan[f"hd95_class{c}"] = hd95_c
            metrics_per_scan[f"nsd_class{c}"] = nsd
            # metrics_per_scan[f"volDiff_class{c}"] = vol_diff_c
            # metrics_per_scan[f"surfDist_class{c}"] = avg_surf_dist_c
            # metrics_per_scan[f"centroidDist_class{c}"] = centroid_dist_c
        all_metrics.append(metrics_per_scan)
    # Create a DataFrame with all per-file metrics
    df_metrics = pd.DataFrame(all_metrics)
    
    # Compute mean and std for each metric (ignoring NaN values by default)
    metric_cols = [c for c in df_metrics.columns if c != "filename"]
    means = df_metrics[metric_cols].mean(numeric_only=True, skipna=True)
    stds  = df_metrics[metric_cols].std(numeric_only=True, skipna=True)
    
    # Build a summary dict
    summary = {}
    for c in range(1, num_classes+1):
        class_summary = {}
        for metric_key in ["dice", "hd95", "nsd"]:
            col_name = f"{metric_key}_class{c}"
            class_summary[metric_key] = {
                "mean": means[col_name],
                "std":  stds[col_name]
            }
        summary[f"class_{class_list[c-1]}"] = class_summary
    
    return df_metrics, summary

if __name__ == "__main__":
    dataset = "Dataset023_BTCV"
    trainer = "dinoUNetTrainer__nnUNetPlans__2d"
    preds_dir = f"/scr/yl_li/segmentation_data/nnUNet_results/{dataset}/{trainer}/fold_0/validation"
    csv_filename = preds_dir + "/segmentation_metrics.csv"

    labels_dir = f"/scr/yl_li/segmentation_data/nnUNet_raw_data_base/nnUNet_raw_data/{dataset}/labelsTr"
    # Load dataset.json which contains the labels mapping (including background)
    dataset_json_path = f"/scr/yl_li/segmentation_data/nnUNet_raw_data_base/nnUNet_raw_data/{dataset}/dataset.json"  # Change this path if needed
    with open(dataset_json_path, "r") as f:
        dataset_info = json.load(f)
    class_list = list(dataset_info['labels'].keys())[1:]
    print(class_list)
    df, summary_dict = compute_segmentation_metrics(preds_dir, labels_dir, class_list)

    aggregated = {metric: {"mean": [], "std": []} for metric in ["dice", "hd95", "nsd"]}
    for class_name, metrics in summary_dict.items():
            for metric_key in aggregated.keys():
                mean_val = metrics[metric_key]["mean"]
                std_val = metrics[metric_key]["std"]
                # Append only if the values are not nan
                if not np.isnan(mean_val):
                    aggregated[metric_key]["mean"].append(mean_val)
                if not np.isnan(std_val):
                    aggregated[metric_key]["std"].append(std_val)
        
    classwise_average = {}
    for metric_key, values in aggregated.items():
        avg_mean = np.mean(values["mean"]) if values["mean"] else np.nan
        avg_std = np.mean(values["std"]) if values["std"] else np.nan
        classwise_average[metric_key] = {"mean": avg_mean, "std": avg_std}

    # Add the aggregated class-wise averages to the renamed summary dict
    summary_dict["classwise_average"] = classwise_average

    # Print the summarized metrics
    print("\nSummary (Mean & Std) per class (NaN entries are ignored in the calculation):")
    for class_label, metrics in summary_dict.items():
        print(f"  {class_label}:")
        for metric_key, stats in metrics.items():
            print(f"    {metric_key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    df.to_csv(csv_filename, index=False)
    print(f"\nDataFrame saved to {csv_filename}")
    