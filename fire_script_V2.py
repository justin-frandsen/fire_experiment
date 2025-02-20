import os
import cv2
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ..gbvs.saliency_models import gbvs, ittikochneibur #set this to path of gbvs folder

if __name__ == '__main__':
    while True:
        graph_var = input("Would you like to graph the images? (y/n): ").lower()
        if graph_var in ['y', 'n']:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    fire_images = os.listdir("fire_images")
    number_imgs = len(fire_images)

    # Create lists to store mean saliency values for all images
    mean_saliency_list = []

    for i in range(1, number_imgs):
        imname = "./fire_images/{}".format(fire_images[i-1])
        print("Processing {}".format(imname))

        img = cv2.imread(imname)

        # Compute saliency maps
        saliency_map_gbvs = gbvs.compute_saliency(img)
        saliency_map_ikn = ittikochneibur.compute_saliency(img)

        if graph_var == 'y':
            # Plot the images
            fig = plt.figure(figsize=(10, 3))

            fig.add_subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
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

        oname = "./fire_output/{}_out{}.jpg".format(fire_images[i-1], time.time())
        cv2.imwrite(oname, saliency_map_gbvs)

        # Save saliency values to a CSV (each pixel's saliency)
        saliency_csv_filename = "./fire_csv/{}_saliency.csv".format(fire_images[i-1])
        np.savetxt(saliency_csv_filename, saliency_map_gbvs, delimiter=",")

        # Compute mean saliency
        mean_saliency = np.mean(saliency_map_gbvs)
        mean_saliency_list.append({"image_name": fire_images[i-1], "mean_saliency": mean_saliency})

    # Save mean saliency values to a CSV
    mean_saliency_df = pd.DataFrame(mean_saliency_list)
    mean_saliency_df.to_csv("./fire_csv/mean_saliency.csv", index=False)

    print("Processing complete. Saliency data saved.")
