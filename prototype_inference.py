#!/usr/bin/env python
"""
prototype_inference.py

Usage:
    python prototype_inference.py \
      --inference_image "/path/to/inference.npy" \
      --model_path "/path/to/best_model.pth" \
      --output_figure "output_figure.png" \
      --save_mask \
      --save_mask_path "predicted_mask.npy" \
      --dilate_mask \
      --dilation_kernel_size "5,5" \
      --dilation_iterations 1 \
      --debug

This script loads a pre-trained segmentation model, runs sliding window inference on a large inference image,
optionally dilates the predicted mask for improved visibility, visualizes the input and predicted mask using
a discrete colormap with a legend, and saves the outputs. Tested on a GPU cluster as well on a MacBook
"""

import os
import time
import argparse
import logging
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import signal
import segmentation_models_pytorch as smp

# Set known fonts to avoid findfont issues.
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Suppress verbose font-manager debug messages.
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Define global variables.
COLORMAP_NAME = "seismic"  # For generating legend (if needed).
# Define class labels (mapping display name to integer index).
CLASS_LABELS = {"Background": 0, "Climbing": 1, "Personnel": 2}
# Define a discrete colormap for mask visualization.
DISCRETE_CMAP = ListedColormap(["black", "red", "green"])

class Prototype:
    def __init__(self, model, model_path, device="cuda", patch_size=(1024, 896), overlap=0.2):
        self.model_path = model_path
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap

        logging.debug(f"Loading model weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        self.model = model.to(device)
        self.model.eval()
        logging.debug(f"Model loaded and set to eval mode on device: {device}")

    @staticmethod
    def _read_image(path):
        """
        Reads an image from disk, applies filtering, and returns a contiguous NumPy array.
        """
        logging.debug(f"Reading image from {path} ...")
        start = time.time()
        data = np.load(path)
        # Example processing: cumulative sum and conversion to float.
        data = np.cumsum(data, axis=0, dtype=np.int64).astype(np.float32)
        data *= np.pi / 2**15  # Convert to radians

        sos = signal.butter(2, 5, "high", fs=1000, output="sos")
        data = signal.sosfiltfilt(sos, data, axis=0)
        image = np.ascontiguousarray(data)
        elapsed = time.time() - start
        logging.debug(f"Image processed in {elapsed:.2f} seconds. Shape: {image.shape}")
        return image

    def run(self, image_path):
        """
        Performs sliding window inference on a large input image.

        Parameters:
          image_path: Path to the .npy image file (expected shape [H, W] for grayscale).

        Returns:
          final_prediction: torch.Tensor of shape [H, W] (with predicted class indices)
        """
        logging.debug(f"Running inference on image: {image_path}")
        overall_start = time.time()

        # Read image from disk.
        image_np = Prototype._read_image(image_path)  # shape: [H, W]
        # Convert to torch.Tensor.
        image = torch.from_numpy(image_np)  # shape: [H, W]
        # If image is 2D (grayscale), add a channel dimension: [1, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
        # Add batch dimension: [1, C, H, W]
        input_tensor = image.unsqueeze(0).float()  # shape: [1, 1, H, W]
        logging.debug(f"Input tensor shape: {input_tensor.shape}")

        # For sliding window inference, work on CPU first.
        image_tensor = input_tensor.cpu()
        C, H, W = image_tensor.shape[1:]
        patch_h, patch_w = self.patch_size
        logging.debug(f"Image dimensions (C, H, W): ({C}, {H}, {W})")
        logging.debug(f"Using patch size: {self.patch_size} with overlap: {self.overlap}")

        stride_h = int(patch_h * (1 - self.overlap))
        stride_w = int(patch_w * (1 - self.overlap))

        full_prediction = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        count_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)

        patch_count = 0
        sliding_start = time.time()
        for top in range(0, H - patch_h + 1, stride_h):
            for left in range(0, W - patch_w + 1, stride_w):
                patch = image_tensor[0, :, top:top+patch_h, left:left+patch_w]
                patch = patch.unsqueeze(0).to(self.device)  # shape: [1, C, patch_h, patch_w]
                with torch.no_grad():
                    output = self.model(patch)  # Expected: [1, num_classes, patch_h, patch_w]
                    pred_patch = torch.argmax(output, dim=1)  # [1, patch_h, patch_w]
                pred_patch = pred_patch[0].float()
                full_prediction[top:top+patch_h, left:left+patch_w] += pred_patch
                count_map[top:top+patch_h, left:left+patch_w] += 1.0
                patch_count += 1
        sliding_elapsed = time.time() - sliding_start
        logging.debug(f"Processed {patch_count} patches in {sliding_elapsed:.2f} seconds.")

        count_map[count_map == 0] = 1.0
        averaged_prediction = full_prediction / count_map
        final_prediction = torch.round(averaged_prediction).to(torch.long)
        overall_elapsed = time.time() - overall_start
        logging.debug(f"Total inference time: {overall_elapsed:.2f} seconds.")
        return final_prediction

def visualize_results(input_image, predicted_mask, class_labels,
                      cmap_input="seismic", cmap_mask="jet", save_path=None,
                      dilate_mask=False, dilation_kernel_size=(5, 5), dilation_iterations=1):
    """
    Visualizes the input image and predicted mask side by side with a legend.
    Optionally applies morphological dilation to the predicted mask for improved visibility.

    Parameters:
      input_image: NumPy array for the input image.
      predicted_mask: NumPy array for the predicted mask (with discrete class indices).
      class_labels: Dictionary mapping class names to integer labels.
      cmap_input: Colormap for the input image.
      cmap_mask: Colormap for the predicted mask.
      save_path: Optional file path to save the figure.
      dilate_mask: If True, apply dilation to the predicted mask.
      dilation_kernel_size: Kernel size for dilation.
      dilation_iterations: Number of dilation iterations.
    """

    import cv2
    logging.debug("Visualizing results...")
    if dilate_mask:
        mask_uint8 = predicted_mask.astype(np.uint8)
        kernel = np.ones(dilation_kernel_size, np.uint8)
        predicted_mask = cv2.dilate(mask_uint8, kernel, iterations=dilation_iterations)
        logging.debug("Applied dilation to predicted mask.")

    # Use a discrete colormap for the mask.
    discrete_cmap = DISCRETE_CMAP
    # Create legend handles.
    legend_patches = [mpatches.Patch(color=discrete_cmap.colors[v], label=k)
                      for k, v in class_labels.items()]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap=cmap_input, aspect="auto")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    img = plt.imshow(predicted_mask, cmap=discrete_cmap, aspect="auto")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.debug(f"Figure saved to {save_path}")
    plt.show()
    logging.debug("Visualization complete.")

def main(args):
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Device selection: check for MPS (Mac Silicon), then CUDA, then CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Set model parameters.
    encoder_name = "resnet50"
    encoder_weights = "imagenet"
    in_channels = 1  # Assuming grayscale images.
    classes = 3

    # Instantiate the model (example: FPN). Adjust if needed.
    model = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )

    # Define patch size and overlap.
    patch_size = (1024, 896)
    overlap = 0.2
    logging.info("Creating Prototype...")
    prototype = Prototype(model, args.model_path, device, patch_size, overlap)

    logging.info("Running inference...")
    start = time.time()
    predicted_mask = prototype.run(args.inference_image)
    elapsed = time.time() - start
    logging.info(f"Inference completed in {elapsed:.2f} seconds.")
    predicted_mask_np = predicted_mask.cpu().numpy()

    # Read input image for visualization.
    input_np = Prototype._read_image(args.inference_image)

    visualize_results(input_np, predicted_mask_np, CLASS_LABELS,
                      cmap_input="seismic", cmap_mask="jet", save_path=args.output_figure,
                      dilate_mask=args.dilate_mask,
                      dilation_kernel_size=tuple(map(int, args.dilation_kernel_size.split(','))),
                      dilation_iterations=args.dilation_iterations)

    if args.save_mask:
        np.save(args.save_mask_path, predicted_mask_np)
        logging.info(f"Predicted mask saved to {args.save_mask_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a large image and visualize results.")
    parser.add_argument("--inference_image", type=str, required=True,
                        help="Path to the inference image (.npy file).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model weights (.pth file).")
    parser.add_argument("--output_figure", type=str, default="output_figure.png",
                        help="Path to save the output figure.")
    parser.add_argument("--save_mask", action="store_true",
                        help="If set, save the predicted mask to a file.")
    parser.add_argument("--save_mask_path", type=str, default="predicted_mask.npy",
                        help="Path to save the predicted mask.")
    parser.add_argument("--dilate_mask", action="store_true",
                        help="If set, apply dilation to the predicted mask for better visibility.")
    parser.add_argument("--dilation_kernel_size", type=str, default="5,5",
                        help="Kernel size for dilation as 'width,height' (e.g., '5,5').")
    parser.add_argument("--dilation_iterations", type=int, default=1,
                        help="Number of dilation iterations.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()
    main(args)
