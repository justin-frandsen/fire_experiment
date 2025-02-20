import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from saliency_models import gbvs, ittikochneibur

class ImageProcessor:
    def __init__(self, graph_var, imname):
        self.graph_var = graph_var
        self.imname = imname
        self.img = cv2.imread(imname)
        self.mean_saliency_list = []

    def process_image(self, counter):
        print(f"Processing {self.imname}, number: {counter}")
        
        # Convert to float to prevent division issues
        img_float = self.img.astype(np.float32) + 1e-6
        blue_channel, green_channel, red_channel = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        
        # Compute normalized color ratios
        red_ratio = red_channel / (red_channel + green_channel + blue_channel)
        green_ratio = green_channel / (red_channel + green_channel + blue_channel)
        
        # Create masks
        fire_mask = (red_ratio > 0.5).astype(np.uint8)  # Fire regions
        vegetation_mask = (green_ratio > 0.5).astype(np.uint8)  # Vegetation regions
        
        # Apply Gaussian blur to reduce noise
        fire_mask = cv2.GaussianBlur(fire_mask, (5, 5), 0)
        vegetation_mask = cv2.GaussianBlur(vegetation_mask, (5, 5), 0)
        
        # Detect contours for ROIs
        fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vegetation_contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create labeled masks for ROI classification
        roi_type_mask = np.zeros_like(fire_mask, dtype=np.uint8)  # 1 = Fire, 2 = Vegetation
        roi_index_mask = np.zeros_like(fire_mask, dtype=np.uint8)  # Unique ID for each ROI
        
        # Assign unique indices to each fire ROI
        for idx, contour in enumerate(fire_contours, start=1):
            cv2.drawContours(roi_type_mask, [contour], -1, 1, thickness=cv2.FILLED)
            cv2.drawContours(roi_index_mask, [contour], -1, idx, thickness=cv2.FILLED)
        
        # Assign unique indices to each vegetation ROI
        for idx, contour in enumerate(vegetation_contours, start=len(fire_contours) + 1):
            cv2.drawContours(roi_type_mask, [contour], -1, 2, thickness=cv2.FILLED)
            cv2.drawContours(roi_index_mask, [contour], -1, idx, thickness=cv2.FILLED)
        
        # Compute saliency maps
        saliency_map_gbvs = gbvs.compute_saliency(self.img)
        
        return fire_mask, vegetation_mask, saliency_map_gbvs, roi_type_mask, roi_index_mask
    
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

if __name__ == '__main__':
    graph_var = input("Would you like to graph the images? (y/n): ").lower()
    images = os.listdir("images")
    
    for counter, image_name in enumerate(images, start=1):
        imname = f"./images/{image_name}"
        processor = ImageProcessor(graph_var, imname)
        fire_mask, vegetation_mask, saliency_map_gbvs, roi_type_mask, roi_index_mask = processor.process_image(counter)
        processor.graphing_func(roi_index_mask, fire_mask, vegetation_mask, saliency_map_gbvs)
        processor.save_contours(roi_type_mask, roi_index_mask, image_name)
