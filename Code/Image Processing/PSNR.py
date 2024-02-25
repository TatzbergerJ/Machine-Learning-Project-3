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
    for filename in os.listdir(folder2):

        filename_no_extension, _ = os.path.splitext(filename)
        db_filename = "original_blurred_" + filename_no_extension + ".jpg"
        print(db_filename)

        og_img_path = os.path.join(folder2, filename)
        #print(og_img_path)
        db_img_path = os.path.join(folder1, db_filename)
        #print(db_img_path)
        # check if the file exists in the second folder
        if not os.path.exists(db_img_path):
            print(f"File {filename} not found in folder {folder1}")
            continue

        psnr = PSNR(og_img_path, db_img_path)
        psnr_values.append(psnr)
        #print(f"PSNR for {filename}: {psnr:.2f} dB")

    return psnr_values

if __name__ == "__main__":


    folder1 = "..\\image-deblurring-using-deep-learning\\test_data\\blurred_same_size_as_output"
    folder2 = "..\\image-deblurring-using-deep-learning\\test_data\\test_deblurred_images_final_results"


    psnr_values = calculate_folder_psnr(folder1, folder2)
    #average_psnr = sum(psnr for _, psnr in psnr_values) / len(psnr_values)
    #print(f"Average PSNR: {average_psnr:.2f} dB")

    if psnr_values:
        average_psnr = sum(psnr for psnr in psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr:.2f} dB")
    else:
        print("No PSNR values calculated. Check input folders.")
