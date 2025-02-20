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
        
        # Draw ROIs on the image
        roi_img = self.img.copy()
        cv2.drawContours(roi_img, fire_contours, -1, (0, 0, 255), 2)  # Red for fire
        cv2.drawContours(roi_img, vegetation_contours, -1, (0, 255, 0), 2)  # Green for vegetation
        
        # Compute saliency maps
        saliency_map_gbvs = gbvs.compute_saliency(self.img)
        saliency_map_ikn = ittikochneibur.compute_saliency(self.img)
        
        return roi_img, fire_mask, vegetation_mask, saliency_map_gbvs, saliency_map_ikn
    
    def graphing_func(self, roi_img, fire_mask, vegetation_mask, saliency_map_gbvs, saliency_map_ikn):
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
        roi_img, fire_mask, vegetation_mask, saliency_map_gbvs, saliency_map_ikn = processor.process_image(counter)
        processor.graphing_func(roi_img, fire_mask, vegetation_mask, saliency_map_gbvs, saliency_map_ikn)