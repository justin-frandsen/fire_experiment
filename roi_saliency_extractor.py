"""
roi_saliency_extractor.py

Improved fire and vegetation region detection and circular ROI removal.

Usage:
    - Place images in the "images" directory.
    - Run the script and follow prompts to enable/disable graphing, saving, and ROI removal.
    - Outputs are saved in "contour_masks", "outputs", and "csv_output" directories.
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
    def __init__(self, show_graphs, postprocess_pref, image_path, top_k=5, radius=20):
        """
        Initialize the processor.

        Args:
            show_graphs (str): 'y' to show graphs, 'n' otherwise.
            image_path (str): Path to the image file.
            top_k (int): Number of largest regions to keep for each ROI type.
        """
        self.show_graphs = show_graphs
        self.postprocess_pref = postprocess_pref
        self.image_path = image_path
        self.top_k = top_k
        self.radius = radius
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
        fire_rgb = (red > np.mean(red) + 0.2 * np.std(red))
        veg_rgb = (green > np.mean(green) + 0.5 * np.std(green))
        fire_hsv, veg_hsv = self._threshold_hsv()
        fire_mask = np.logical_or(fire_rgb, fire_hsv).astype(np.uint8)
        veg_mask = np.logical_or(veg_rgb, veg_hsv).astype(np.uint8)
        return fire_mask, veg_mask

    def _threshold_hsv(self):
        """
        Utilize HSV color space to create masks for fire and vegetation.
        This is a more robust method for detecting these regions because
        of the brightness and color characteristics of fire.
        """
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        fire_mask = ((h < 30) & (s > 100) & (v > 150)).astype(np.uint8)
        veg_mask = ((h > 35) & (h < 85) & (s > 50) & (v > 50)).astype(np.uint8)
        return fire_mask, veg_mask

    def _binarize_masks(self, fire, veg):
        """
        Smooth and binarize the fire and vegetation masks.
        """
        kernel = np.ones((5, 5), np.uint8)
        fire_clean = cv2.morphologyEx(fire, cv2.MORPH_OPEN, kernel)
        veg_clean = cv2.morphologyEx(veg, cv2.MORPH_OPEN, kernel)
        return fire_clean, veg_clean

    def _extract_contours(self, fire_mask, veg_mask):
        """
        Extract contours from the binary masks.
        """
        fire_ctrs, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        veg_ctrs, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return fire_ctrs, veg_ctrs

    def _keep_largest_regions(self, fire_ctrs, veg_ctrs, shape):
        """
        Keep only the top-k largest fire and vegetation regions based on contour area.
        Create blank masks and draw the selected contours onto them.
        Note: cv2.drawContours modifies the mask arrays in-place.
        """
        # Sort and keep the top-k largest contours by area
        fire_ctrs = sorted(fire_ctrs, key=cv2.contourArea, reverse=True)[:self.top_k]
        veg_ctrs = sorted(veg_ctrs, key=cv2.contourArea, reverse=True)[:self.top_k]
        
        # Create empty masks with the given shape
        fire_mask = np.zeros(shape, dtype=np.uint8)
        veg_mask = np.zeros(shape, dtype=np.uint8)
        
        # Draw the top-k contours onto the masks (modifies masks in-place)
        cv2.drawContours(fire_mask, fire_ctrs, -1, 255, -1)
        cv2.drawContours(veg_mask, veg_ctrs, -1, 255, -1)
        
        # Return the binary masks and the filtered contours
        return fire_mask, veg_mask, fire_ctrs, veg_ctrs

    def _label_rois(self, fire_ctrs, veg_ctrs, shape):
        """
        Create masks for ROI types and indices based on contours.
        Fire regions are labeled as 1, vegetation as 2 and background as 0.
        Each ROI index is unique and starts from 1.
        """
        # Initialize blank masks:
        # roi_type: 0 = background, 1 = fire, 2 = vegetation
        # roi_index: unique ID per region, starting from 1
        roi_type = np.zeros(shape, dtype=np.uint8)
        roi_index = np.zeros(shape, dtype=np.uint16)
        # Label fire regions
        for i, ctr in enumerate(fire_ctrs):
            cv2.drawContours(roi_type, [ctr], -1, 1, -1) # Type 1 = fire
            cv2.drawContours(roi_index, [ctr], -1, i + 1, -1) # Unique ID
        
        # Offset vegetation IDs so they don’t overlap with fire
        offset = len(fire_ctrs)
        for i, ctr in enumerate(veg_ctrs):
            cv2.drawContours(roi_type, [ctr], -1, 2, -1) # Type 2 = vegetation
            cv2.drawContours(roi_index, [ctr], -1, offset + i + 1, -1) # Unique ID
            
        # Return both labeled masks
        return roi_type, roi_index

    def _store_masks(self, fire_ctrs, veg_ctrs, fire_mask, veg_mask):
        """
        Store masks and contours as attributes for later use.
        """
        self.fire_contours = fire_ctrs
        self.veg_contours = veg_ctrs
        self.fire_mask_bin = fire_mask
        self.veg_mask_bin = veg_mask

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

    def postprocess_rois(self):
        """
        Allow user to click on ROIs to remove them interactively.
        Overlays the ROI index mask (multicolor) on the original image.
        Removes the entire ROI (all pixels with the same ROI index) from both roi_index_mask and roi_type_mask.
        """
        if self.postprocess_pref != 'y':
            return

        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_rgb)
        overlay = ax.imshow(self.roi_index_mask, cmap='nipy_spectral', alpha=0.5)
        ax.set_title("Press 'e' to erase, 'g' to grow, or 'd' to delete. Click and drag. Close window when done.")
        plt.axis('off')

        self.mode = 'e'  # default mode
        self.dragging = False
        self.current_roi_id = None

        def apply_circle_mask(x, y):
            yy, xx = np.ogrid[:self.roi_index_mask.shape[0], :self.roi_index_mask.shape[1]]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= self.radius ** 2
            return mask

        def on_key(event):
            if event.key == 'e':
                self.mode = 'e'
                print("Mode: Erase")
            elif event.key == 'g':
                self.mode = 'g'
                print("Mode: Grow")
            elif event.key == 'd':
                self.mode = 'd'
                print("Mode: Delete")
            elif event.key == '=':
                self.mode = '='
                self.radius = self.radius + 5
                print(f"Radius set to {self.radius}")
            elif event.key == '-':
                self.mode = '-'
                self.radius = self.radius - 5
                print(f"Radius set to {self.radius}")
            elif event.key == '0':
                self.mode = '0'
                print("Mode: Type checker")


        def on_press(event):
            if event.xdata is None or event.ydata is None:
                return
            self.dragging = True
            x, y = int(event.xdata), int(event.ydata)

            if self.mode == 'e':
                mask = apply_circle_mask(x, y)
                print(f"Erasing at ({x},{y})")
                self.roi_index_mask[mask] = 0
                self.roi_type_mask[mask] = 0
            elif self.mode == 'g':
                roi_id = self.roi_index_mask[y, x]
                if roi_id == 0:
                    print("Clicked background. Nothing to grow.")
                    self.dragging = False
                    return
                self.current_roi_id = roi_id
                mask = apply_circle_mask(x, y)
                print(f"Growing ROI {roi_id} at ({x},{y})")
                self.roi_index_mask[mask] = roi_id
                roi_type = 1 if roi_id <= len(self.fire_contours) else 2
                self.roi_type_mask[mask] = roi_type
            elif self.mode == 'd':
                x, y = int(event.xdata), int(event.ydata)
                roi_idx = self.roi_index_mask[y, x]
                if roi_idx == 0:
                    print("Clicked background, nothing to remove.")
                    return
                print(f"Removing ROI type: {self.roi_type_mask[y, x]} at ({x},{y})")
                print(f"Removing ROI index: {roi_idx}")
                # Remove all pixels with this ROI index from both masks
                self.roi_index_mask[self.roi_index_mask == roi_idx] = 0
                self.roi_type_mask[self.roi_index_mask == 0] = 0
                overlay.set_data(self.roi_index_mask)
                fig.canvas.draw_idle()
            elif self.mode == '0':
                print(f"ROI type: {self.roi_type_mask[y, x]} at ({x},{y})")

            overlay.set_data(self.roi_index_mask)
            fig.canvas.draw_idle()

        def on_motion(event):
            if not self.dragging or event.xdata is None or event.ydata is None:
                return
            x, y = int(event.xdata), int(event.ydata)

            if self.mode == 'e':
                mask = apply_circle_mask(x, y)
                self.roi_index_mask[mask] = 0
                self.roi_type_mask[mask] = 0
            elif self.mode == 'g' and self.current_roi_id is not None:
                mask = apply_circle_mask(x, y)
                self.roi_index_mask[mask] = self.current_roi_id
                roi_type = 1 if self.current_roi_id <= len(self.fire_contours) else 2
                self.roi_type_mask[mask] = roi_type

            overlay.set_data(self.roi_index_mask)
            fig.canvas.draw_idle()

        def on_release(event):
            self.dragging = False
            self.current_roi_id = None

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        plt.show()


    def save_visuals(self, image_name):
        """
        Save ROI masks as .npy files in the contour_masks directory.
        """
        os.makedirs("./contour_masks", exist_ok=True)
        np.save(f"./contour_masks/{image_name}_roi_type.npy", self.roi_type_mask)
        np.save(f"./contour_masks/{image_name}_roi_index.npy", self.roi_index_mask)
        print(f"Saved ROI masks for {image_name}")

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
    postprocess_pref = input("Do you want to click to remove circular ROI areas? (y/n): ").strip().lower()
    save_pref = input("Save results? (y/n): ").strip().lower()

    # Prompt for circular ROI removal radius
    try:
        radius = int(input("Enter circular ROI removal radius (e.g., 20): ").strip())
    except:
        radius = 20

    # List all images in the images directory
    images = [f for f in os.listdir("images") if not f.startswith(".")]

    #loop through images and process each one
    for idx, img_name in enumerate(images, 1):
        processor = ImageProcessor(graph_pref, postprocess_pref, os.path.join("images", img_name), radius=radius)
        processor.process_image(idx)

        if graph_pref == 'y':
            plot_roi_index_map(processor.roi_index_mask, f"ROI Index - {img_name}")
            processor.visualize()

        if postprocess_pref == 'y':
            processor.postprocess_rois()

        if save_pref == 'y':
            processor.save_visuals(img_name)
            processor.save_saliency_outputs()

    if save_pref == 'y':
        processor.save_mean_saliency()
