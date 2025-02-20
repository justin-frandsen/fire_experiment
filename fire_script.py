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

    def graphing_func(self, blue_channel, green_channel, red_channel, saliency_map_gbvs, saliency_map_ikn):
        """
        Plots the original image, saliency maps, and color channels.

        Parameters:
        blue_channel (numpy.ndarray): The blue channel of the image.
        green_channel (numpy.ndarray): The green channel of the image.
        red_channel_masked (numpy.ndarray): The masked red channel of the image.
        saliency_map_gbvs (numpy.ndarray): The saliency map computed using GBVS.
        saliency_map_ikn (numpy.ndarray): The saliency map computed using Itti Koch Neibur.
        """
        if self.graph_var == 'y':
            # Plot the images
            fig = plt.figure(figsize=(10, 3))

            fig.add_subplot(1, 3, 1)
            plt.imshow(self.img, cmap='gray')
            plt.gca().set_title("Original Image")
            plt.axis('off')

            fig.add_subplot(1, 3, 2)
            plt.imshow(saliency_map_gbvs, cmap='gray')
            plt.gca().set_title("GBVS")
            plt.axis('off')

            fig.add_subplot(1, 3, 3)
            plt.imshow(saliency_map_ikn, cmap='gray')
            plt.gca().set_title("Itti Koch Neibur")
            plt.axis('off')

            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(blue_channel, cmap='gray')
            axs[0].set_title('Blue Channel')
            axs[0].axis('off')

            axs[1].imshow(green_channel, cmap='gray')
            axs[1].set_title('Green Channel')
            axs[1].axis('off')

            axs[2].imshow(red_channel, cmap='gray')
            axs[2].set_title('Red Channel')
            axs[2].axis('off')

            plt.show()

    def process_image(self, counter):
        print(f"Processing {self.imname}, number: {counter}")

        # Access color channels
        blue_channel = self.img[:, :, 0]
        green_channel = self.img[:, :, 1]
        red_channel = self.img[:, :, 2]

        # Create a mask where red channel values are over 100
        red_mask = red_channel > 200

        # Apply the mask to the red channel
        red_channel_masked = np.zeros_like(red_channel)
        red_channel_masked[red_mask] = red_channel[red_mask]

        # Create a mask where red channel values are over 100
        green_mask = green_channel > 200

        # Apply the mask to the red channel
        green_channel_masked = np.zeros_like(green_channel)
        green_channel_masked[green_mask] = green_channel[green_mask]

        # Compute saliency maps
        saliency_map_gbvs = gbvs.compute_saliency(self.img)
        saliency_map_ikn = ittikochneibur.compute_saliency(self.img)

        return blue_channel, green_channel_masked, red_channel_masked, saliency_map_gbvs, saliency_map_ikn

    def save_image(self, saliency_map_gbvs):
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
    while True:
        graph_var = input("Would you like to graph the images? (y/n): ").lower()
        if graph_var in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    images = os.listdir("images")
    number_imgs = len(images)

    for counter, image_name in enumerate(images, start=1):
        imname = f"./images/{image_name}"
        processor = ImageProcessor(graph_var, imname)
        
        [blue_channel, 
         green_channel_masked, 
         red_channel_masked, 
         saliency_map_gbvs, 
         saliency_map_ikn] = processor.process_image(counter)
        
        processor.graphing_func(blue_channel, green_channel_masked, red_channel_masked, saliency_map_gbvs, saliency_map_ikn)
        
        #processor.save_image(gbvs.compute_saliency(processor.img))

    processor.save_mean_saliency()