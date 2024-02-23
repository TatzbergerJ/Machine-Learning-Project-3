import numpy as np
import os
import cv2
from math import log10, sqrt


def PSNR(og_img_path: str, db_img_path: str):
    """
    reads in paths of an original image and a deblurred version of the image and returns the PSNR value (in dB) in order
    to compare the pictures
    :param og_img_path: path of the original image
    :param db_img_path: path of the deblurred image
    :return: PSNR ratio value (in dB)
    """
    try:
        og_img = cv2.imread(og_img_path)
        db_img = cv2.imread(db_img_path)
    except Exception as e:
        print(f"Error reading images: {e}")
        return float("NaN")

    # check if both images are of the same dimensions
    #height_og, width_og, _ = og_img.shape
    #height_db, width_db, _ = db_img.shape
    #ratio_og = height_og/width_og
    #ratio_db = height_db/width_db

    # check if the pictures have the same ratio and size, if not: return a message and NaN
    if og_img is None or db_img is None:
        print("Error: One or both images could not be read.")
        return float("NaN")

    if og_img.shape != db_img.shape:
        print("Error: Images are not of the same size.")
        return float("NaN")

    mse = np.mean((og_img-db_img)**2)
    if mse == 0: # if both pictures are exactly the same
        return 100
    max_pixel = 255.0
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr

def calculate_folder_psnr(folder1: str, folder2: str):
    psnr_values = []
    for filename in os.listdir(folder1):

        filename_no_extension, _ = os.path.splitext(filename)
        db_filename = filename_no_extension + ".png"

        og_img_path = os.path.join(folder1, filename)
        db_img_path = os.path.join(folder2, db_filename)

        # check if the file exists in the second folder
        if not os.path.exists(db_img_path):
            print(f"File {filename} not found in folder {folder2}")
            continue

        psnr = PSNR(og_img_path, db_img_path)
        psnr_values.append(psnr)
        #print(f"PSNR for {filename}: {psnr:.2f} dB")

    return psnr_values

if __name__ == "__main__":


    folder1 = "coco_images/blurred_images_gaussian"
    folder2 = "coco_images/output_gopro_l1_gaussian"


    psnr_values = calculate_folder_psnr(folder1, folder2)
    #average_psnr = sum(psnr for _, psnr in psnr_values) / len(psnr_values)
    #print(f"Average PSNR: {average_psnr:.2f} dB")

    if psnr_values:
        average_psnr = sum(psnr for psnr in psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr:.2f} dB")
    else:
        print("No PSNR values calculated. Check input folders.")
