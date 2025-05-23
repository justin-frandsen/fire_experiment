import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from saliency_models import gbvs

def plot_roi_index_map(roi_index_mask, title="ROI Index Map"):
    plt.figure(figsize=(8, 6))
    plt.imshow(roi_index_mask, cmap='nipy_spectral')  # 'nipy_spectral' gives a unique color per ID
    plt.colorbar(label="Region Index")
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


class ImageProcessor:
    def __init__(self, graph_var, imname, top_k=5):
        self.graph_var = graph_var
        self.imname = imname
        self.top_k = top_k
        self.img = cv2.imread(imname)
        self.mean_saliency_list = []

    def process_image(self, counter):
        print(f"Processing {self.imname}, number: {counter}")
        img_float, blue_channel, green_channel, red_channel = self._prepare_image()
        red_ratio, green_ratio = self._compute_color_ratios(red_channel, green_channel, blue_channel)
        fire_mask, vegetation_mask = self._compute_masks(red_ratio, green_ratio)
        fire_mask_bin, vegetation_mask_bin = self._postprocess_masks(fire_mask, vegetation_mask)
        fire_contours, vegetation_contours = self._find_contours(fire_mask_bin, vegetation_mask_bin)
        roi_type_mask, roi_index_mask = self._assign_rois(fire_contours, vegetation_contours, fire_mask_bin.shape)
        self._save_results(
            fire_contours, vegetation_contours, roi_type_mask, roi_index_mask,
            fire_mask_bin, vegetation_mask_bin
        )
        self.saliency_map_gbvs = gbvs.compute_saliency(self.img)

    def _prepare_image(self):
        img_float = self.img.astype(np.float32) + 1e-6
        blue_channel = img_float[:, :, 0]
        green_channel = img_float[:, :, 1]
        red_channel = img_float[:, :, 2]
        return img_float, blue_channel, green_channel, red_channel

    def _compute_color_ratios(self, red_channel, green_channel, blue_channel):
        total = red_channel + green_channel + blue_channel
        red_ratio = red_channel / total
        green_ratio = green_channel / total
        return red_ratio, green_ratio

    def _compute_masks(self, red_ratio, green_ratio):
        red_mean = np.mean(red_ratio)
        red_std = np.std(red_ratio)
        adaptive_red_thresh = red_mean + 0.5 * red_std
        fire_mask = (red_ratio > adaptive_red_thresh).astype(np.uint8)

        green_mean = np.mean(green_ratio)
        green_std = np.std(green_ratio)
        adaptive_green_thresh = green_mean + 0.5 * green_std
        vegetation_mask = (green_ratio > adaptive_green_thresh).astype(np.uint8)
        return fire_mask, vegetation_mask

    def _postprocess_masks(self, fire_mask, vegetation_mask):
        fire_mask_blur = cv2.GaussianBlur(fire_mask, (5, 5), 0)
        vegetation_mask_blur = cv2.GaussianBlur(vegetation_mask, (5, 5), 0)
        _, fire_mask_bin = cv2.threshold(fire_mask_blur, 0.5, 1, cv2.THRESH_BINARY)
        _, vegetation_mask_bin = cv2.threshold(vegetation_mask_blur, 0.5, 1, cv2.THRESH_BINARY)
        fire_mask_bin = (fire_mask_bin * 255).astype(np.uint8)
        vegetation_mask_bin = (vegetation_mask_bin * 255).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        fire_mask_bin = cv2.morphologyEx(fire_mask_bin, cv2.MORPH_CLOSE, kernel)
        vegetation_mask_bin = cv2.morphologyEx(vegetation_mask_bin, cv2.MORPH_CLOSE, kernel)
        return fire_mask_bin, vegetation_mask_bin

    def _find_contours(self, fire_mask_bin, vegetation_mask_bin):
        fire_contours, _ = cv2.findContours(fire_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vegetation_contours, _ = cv2.findContours(vegetation_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return fire_contours, vegetation_contours

    def _assign_rois(self, fire_contours, vegetation_contours, mask_shape):
        roi_type_mask = np.zeros(mask_shape, dtype=np.uint8)
        roi_index_mask = np.zeros(mask_shape, dtype=np.uint16)
        for i, contour in enumerate(fire_contours):
            cv2.drawContours(roi_type_mask, [contour], -1, color=1, thickness=-1)
            cv2.drawContours(roi_index_mask, [contour], -1, color=i+1, thickness=-1)
        offset = len(fire_contours)
        for i, contour in enumerate(vegetation_contours):
            cv2.drawContours(roi_type_mask, [contour], -1, color=2, thickness=-1)
            cv2.drawContours(roi_index_mask, [contour], -1, color=offset+i+1, thickness=-1)
        return roi_type_mask, roi_index_mask

    def _save_results(self, fire_contours, vegetation_contours, roi_type_mask, roi_index_mask, fire_mask_bin, vegetation_mask_bin):
        self.fire_contours = fire_contours
        self.vegetation_contours = vegetation_contours
        self.roi_type_mask = roi_type_mask
        self.roi_index_mask = roi_index_mask
        self.fire_mask_bin = fire_mask_bin
        self.vegetation_mask_bin = vegetation_mask_bin

    def save_contours(self, roi_type_mask, roi_index_mask, image_name):
        os.makedirs("./contour_masks", exist_ok=True)
        np.save(f"./contour_masks/{image_name}_roi_type.npy", roi_type_mask)
        np.save(f"./contour_masks/{image_name}_roi_index.npy", roi_index_mask)
        print(f"Saved ROI type and index masks for {image_name}")
    
    def graphing_func(self, roi_img, fire_mask, vegetation_mask, saliency_map_gbvs):
        """
        Displays the original image, fire mask, vegetation mask, GBVS saliency map, and ROIs highlighted.

        Parameters:
        roi_img (numpy.ndarray): Image with ROIs highlighted.
        fire_mask (numpy.ndarray): Binary mask indicating fire regions.
        vegetation_mask (numpy.ndarray): Binary mask indicating vegetation regions.
        saliency_map_gbvs (numpy.ndarray): Saliency map computed using GBVS.
        """
        if self.graph_var == 'y':
            fig, axs = plt.subplots(1, 5, figsize=(20, 5))
            
            titles = ['Original Image', 'Fire Mask', 'Vegetation Mask', 'GBVS Saliency', 'ROIs Highlighted']
            images = [cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB), fire_mask, vegetation_mask, saliency_map_gbvs, roi_img]
            cmaps = [None, 'hot', 'Greens', 'gray', None]
            
            for ax, title, img, cmap in zip(axs, titles, images, cmaps):
                if cmap:
                    ax.imshow(img, cmap=cmap)
                else:
                    ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            plt.show()

    def save_image(self, saliency_map_gbvs):
        #create directories if they don't exist
        os.makedirs("./csv_output", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        # Save the saliency map image
        output_image_name = f"./outputs/{os.path.basename(self.imname)}_out{time.time()}.jpg"
        cv2.imwrite(output_image_name, saliency_map_gbvs)

        # Save saliency values to a CSV
        saliency_csv_filename = f"./csv_output/{os.path.basename(self.imname)}_saliency.csv"
        np.savetxt(saliency_csv_filename, saliency_map_gbvs, delimiter=",")

        # Compute mean saliency
        mean_saliency = np.mean(saliency_map_gbvs)
        self.mean_saliency_list.append({"image_name": os.path.basename(self.imname), "mean_saliency": mean_saliency})

    def save_mean_saliency(self):
        # Save mean saliency values to a CSV
        mean_saliency_df = pd.DataFrame(self.mean_saliency_list)
        mean_saliency_df.to_csv("./csv_output/mean_saliency.csv", index=False)
        print("Processing complete. Saliency data saved.")

if __name__ == '__main__':
    graph_var = input("Would you like to graph the images? (y/n): ").lower()
    images = os.listdir("images")

    for counter, image_name in enumerate(images, start=1):
        imname = f"./images/{image_name}"
        processor = ImageProcessor(graph_var, imname)

        # Just call the function — no need to unpack anything
        processor.process_image(counter)

        # Plot ROI index map
        plot_roi_index_map(processor.roi_index_mask, title=f"ROI Index Map - {image_name}")

        # Show all relevant visualizations
        processor.graphing_func(
            roi_img=processor.roi_index_mask,
            fire_mask=processor.fire_mask_bin,
            vegetation_mask=processor.vegetation_mask_bin,
            saliency_map_gbvs=processor.saliency_map_gbvs
        )

        # Optional: save results
        # processor.save_contours(processor.roi_type_mask, processor.roi_index_mask, image_name)
        # processor.save_image(processor.saliency_map_gbvs)

    # Optional: save saliency summary
    # processor.save_mean_saliency()

