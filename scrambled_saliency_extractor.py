# current script for getting the saliency maps for the scrambled images
# currently we will utilize the same rois as the fire images
import os
import cv2
import time
import numpy as np
from saliency_models import gbvs

def save_saliency_outputs(image_path, saliency_map_gbvs):
        """
        Save the saliency map as an image and CSV, and record mean saliency.
        """
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./csv_output", exist_ok=True)

        ts = int(time.time())
        base_name = os.path.basename(image_path)
        out_img_path = f"./outputs/{base_name}_out{ts}.jpg"
        out_csv_path = f"./csv_output/{base_name}_saliency.csv"

        cv2.imwrite(out_img_path, saliency_map_gbvs)
        np.savetxt(out_csv_path, saliency_map_gbvs, delimiter=',')

# List all images
images = [f for f in os.listdir("not_used") if not f.startswith(".")]

for idx, img_name in enumerate(images, 1):
    print(img_name)
    image = cv2.imread(f"./not_used/{img_name}")
    saliency_map_gbvs = gbvs.compute_saliency(image)
    save_saliency_outputs(img_name, saliency_map_gbvs)
    