"""
roi_saliency_extractor.py

This script processes images to extract region-of-interest (ROI) masks and saliency maps for fire and vegetation regions.
It uses color ratios to segment fire and vegetation, extracts contours, labels ROIs, and computes saliency maps using GBVS.
The results (masks and saliency maps) can be saved for downstream analysis, and visualizations are available for inspection.

Usage:
    - Place images in the "images" directory.
    - Run the script and follow prompts to enable/disable graphing and saving.
    - Outputs are saved in "contour_masks", "outputs", and "csv_output" directories.

Dependencies:
    - OpenCV
    - NumPy
    - Pandas
    - Matplotlib
    - saliency_models.gbvs (custom or external module)
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from saliency_models import gbvs

def plot_roi_index_map(mask, title="ROI Index Map"):
    """
    Plot the ROI index mask using a color map for visualization.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='nipy_spectral')
    plt.colorbar(label="Region Index")
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


class ImageProcessor:
    """
    Processes a single image to extract fire and vegetation ROIs and compute a saliency map.
    """
    def __init__(self, show_graphs, image_path, top_k=5):
        """
        Initialize the processor.

        Args:
            show_graphs (str): 'y' to show graphs, 'n' otherwise.
            image_path (str): Path to the image file.
            top_k (int): Number of largest regions to keep for each ROI type.
        """
        self.show_graphs = show_graphs
        self.image_path = image_path
        self.top_k = top_k
        self.img = cv2.imread(image_path)
        self.mean_saliency_list = []

    def process_image(self, idx):
        """
        Main processing pipeline for an image.
        """
        print(f"Processing {self.image_path}, image #{idx}")

        img_f32, b, g, r = self._prepare_image()
        red_ratio, green_ratio = self._compute_color_ratios(r, g, b)
        fire_mask, veg_mask = self._threshold_ratios(red_ratio, green_ratio)
        fire_bin, veg_bin = self._binarize_masks(fire_mask, veg_mask)
        fire_ctrs, veg_ctrs = self._extract_contours(fire_bin, veg_bin)

        fire_mask_largest, veg_mask_largest, fire_ctrs, veg_ctrs = self._keep_largest_regions(
            fire_ctrs, veg_ctrs, fire_bin.shape
        )

        self.roi_type_mask, self.roi_index_mask = self._label_rois(
            fire_ctrs, veg_ctrs, fire_bin.shape
        )

        self._store_masks(fire_ctrs, veg_ctrs, fire_bin, veg_bin)
        self.saliency_map_gbvs = gbvs.compute_saliency(self.img)

    def _prepare_image(self):
        """
        Convert image to float32 and split into B, G, R channels.
        """
        img = self.img.astype(np.float32) + 1e-6
        return img, img[:, :, 0], img[:, :, 1], img[:, :, 2]

    def _compute_color_ratios(self, r, g, b):
        """
        Compute red and green color ratios for segmentation.
        """
        total = r + g + b
        return r / total, g / total

    def _threshold_ratios(self, red, green):
        """
        Threshold color ratios to create binary masks for fire and vegetation.
        """
        fire_thresh = np.mean(red) + 0.2 * np.std(red)
        veg_thresh = np.mean(green) + 0.5 * np.std(green)
        return (red > fire_thresh).astype(np.uint8), (green > veg_thresh).astype(np.uint8)

    def _binarize_masks(self, fire, veg):
        """
        Smooth and binarize the fire and vegetation masks.
        """
        fire_blur = cv2.GaussianBlur(fire, (5, 5), 0)
        veg_blur = cv2.GaussianBlur(veg, (5, 5), 0)
        _, fire_bin = cv2.threshold(fire_blur, 0.5, 255, cv2.THRESH_BINARY)
        _, veg_bin = cv2.threshold(veg_blur, 0.5, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        return cv2.morphologyEx(fire_bin, cv2.MORPH_CLOSE, kernel), cv2.morphologyEx(veg_bin, cv2.MORPH_CLOSE, kernel)

    def _extract_contours(self, fire_mask, veg_mask):
        """
        Extract contours from the binary masks.
        """
        fire_ctrs, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        veg_ctrs, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return fire_ctrs, veg_ctrs

    def _label_rois(self, fire_ctrs, veg_ctrs, shape):
        """
        Create ROI type and index masks from contours.
        """
        roi_type = np.zeros(shape, dtype=np.uint8)
        roi_index = np.zeros(shape, dtype=np.uint16)
        # Label fire regions as 1, vegetation as 2
        for i, ctr in enumerate(fire_ctrs):
            cv2.drawContours(roi_type, [ctr], -1, 1, -1)
            cv2.drawContours(roi_index, [ctr], -1, i+1, -1)
        offset = len(fire_ctrs)
        for i, ctr in enumerate(veg_ctrs):
            cv2.drawContours(roi_type, [ctr], -1, 2, -1)
            cv2.drawContours(roi_index, [ctr], -1, offset + i + 1, -1)
        return roi_type, roi_index

    def _store_masks(self, fire_ctrs, veg_ctrs, fire_mask, veg_mask):
        """
        Store masks and contours as attributes for later use.
        """
        self.fire_contours = fire_ctrs
        self.veg_contours = veg_ctrs
        self.fire_mask_bin = fire_mask
        self.veg_mask_bin = veg_mask

    def _keep_largest_regions(self, fire_ctrs, veg_ctrs, shape):
        """
        Keep only the largest top_k fire and vegetation regions.
        """
        fire_ctrs = sorted(fire_ctrs, key=cv2.contourArea, reverse=True)[:self.top_k]
        veg_ctrs = sorted(veg_ctrs, key=cv2.contourArea, reverse=True)[:self.top_k]
        fire_mask = np.zeros(shape, dtype=np.uint8)
        veg_mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(fire_mask, fire_ctrs, -1, 255, -1)
        cv2.drawContours(veg_mask, veg_ctrs, -1, 255, -1)
        return fire_mask, veg_mask, fire_ctrs, veg_ctrs

    def save_visuals(self, image_name):
        """
        Save ROI masks as .npy files in the contour_masks directory.
        """
        os.makedirs("./contour_masks", exist_ok=True)
        np.save(f"./contour_masks/{image_name}_roi_type.npy", self.roi_type_mask)
        np.save(f"./contour_masks/{image_name}_roi_index.npy", self.roi_index_mask)
        print(f"Saved ROI masks for {image_name}")

    def visualize(self):
        """
        Show visualizations of the original image, masks, saliency, and ROI map.
        """
        if self.show_graphs != 'y':
            return
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))
        titles = ['Original', 'Fire Mask', 'Vegetation Mask', 'Saliency Map', 'ROI Map']
        images = [cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), self.fire_mask_bin, self.veg_mask_bin,
                  self.saliency_map_gbvs, self.roi_index_mask]
        cmaps = [None, 'hot', 'Greens', 'gray', None]
        for ax, title, img, cmap in zip(axs, titles, images, cmaps):
            ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def save_saliency_outputs(self):
        """
        Save the saliency map as an image and CSV, and record mean saliency.
        """
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./csv_output", exist_ok=True)

        ts = int(time.time())
        base_name = os.path.basename(self.image_path)
        out_img_path = f"./outputs/{base_name}_out{ts}.jpg"
        out_csv_path = f"./csv_output/{base_name}_saliency.csv"

        cv2.imwrite(out_img_path, self.saliency_map_gbvs)
        np.savetxt(out_csv_path, self.saliency_map_gbvs, delimiter=',')
        self.mean_saliency_list.append({
            "image_name": base_name,
            "mean_saliency": np.mean(self.saliency_map_gbvs)
        })

    def save_mean_saliency(self):
        """
        Save a CSV summary of mean saliency values for all processed images.
        """
        df = pd.DataFrame(self.mean_saliency_list)
        df.to_csv("./csv_output/mean_saliency.csv", index=False)
        print("Saved mean saliency summary.")


if __name__ == '__main__':
    # Prompt user for preferences
    graph_pref = input("Graph images? (y/n): ").strip().lower()
    save_pref = input("Save results? (y/n): ").strip().lower()

    # List all images in the images directory
    images = [f for f in os.listdir("images") if not f.startswith(".")]

    for idx, img_name in enumerate(images, 1):
        processor = ImageProcessor(graph_pref, os.path.join("images", img_name))
        processor.process_image(idx)

        if graph_pref == 'y':
            plot_roi_index_map(processor.roi_index_mask, f"ROI Index - {img_name}")
            processor.visualize()

        if save_pref == 'y':
            processor.save_visuals(img_name)
            processor.save_saliency_outputs()

    if save_pref == 'y':
        processor.save_mean_saliency()