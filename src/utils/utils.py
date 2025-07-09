import os
import numpy as np
import torch
import json
from easydict import EasyDict
from torchvision.ops import nms
from matplotlib import pyplot as plt


def nms_on_predictions(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) on the predictions.
    Args:
        predictions (list): List of predictions, each containing 'boxes', 'scores', and 'labels'.
        iou_threshold (float): IoU threshold for NMS.
    Returns:
        list: Filtered predictions after applying NMS.
    """
    filtered_predictions = []
    
    for pred in predictions:
        boxes = torch.tensor(pred['boxes'])
        scores = torch.tensor(pred['scores'])

        keep_indices = nms(boxes, scores, iou_threshold)
        
        filtered_boxes = pred['boxes'][keep_indices]
        filtered_scores = pred['scores'][keep_indices]
        filtered_labels = pred['labels'][keep_indices]

        filtered_predictions.append({
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels
        })
    
    return filtered_predictions

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.
    Args:
        batch (list): List of tuples, where each tuple contains an image and its corresponding target.
    Returns:
        tuple: A tuple containing a list of images and a list of targets.
    """
    images, targets = zip(*batch)  # Unzip
    return list(images), list(targets)

def check_image_format(image):
    """
    Ensure the image is in the correct format for processing.

    Args:
        image (np.ndarray): The image to check. Shape: (C, H, W) or (BATCH, C, H, W).

    Returns:
        np.ndarray: The image in the correct format. Shape: (BATCH, C, H, W).
    """
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    return image


def load_config(cfg_path: str) -> EasyDict:
    """Loads a config json and returns a edict object."""
    with open(cfg_path, "r", encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)

    return EasyDict(cfg_dict)

def plot_patch(patch, file_name, folder="data/"):
    """
    Plot the adversarial patch.

    Args:
        patch (np.ndarray): The adversarial patch to plot.
        file_name (str): The name of the file to save the plot. Without the .png at the end.
    """
    os.makedirs(folder, exist_ok=True)
    
    plt.figure()
    plt.axis("off")
    plt.title(file_name)
    plt.imshow(patch.transpose(1, 2, 0).astype(np.uint8))
    plt.savefig(folder + file_name + ".png")
    plt.show()
    plt.close()