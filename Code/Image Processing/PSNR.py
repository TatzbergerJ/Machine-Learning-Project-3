import numpy as np
import os
import cv2
from math import log10, sqrt
import re
from typing import List


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


def calculate_folder_psnr(folder1: str, folder2: str) -> List[float]:
    psnr_values = []
    # Regular expression to extract image numbers
    pattern = re.compile(r'_(\d+)\.jpg')

    # Iterate over files in the first folder
    for filename1 in os.listdir(folder1):
        # Extract image number from filename in folder1
        match1 = pattern.search(filename1)
        if match1:
            image_number1 = match1.group(1)  # Extract the first group (image number)
            # Construct corresponding filename in folder2
            filename2 = f"sharp_COCO_train2014_{image_number1}.jpg"
            image_path2 = os.path.join(folder2, filename2)
            # Check if the file exists in folder2
            if os.path.exists(image_path2):
                # Calculate PSNR and append to list
                psnr = PSNR(os.path.join(folder1, filename1), image_path2)
                psnr_values.append(psnr)
            else:
                print(f"File {filename2} not found in folder {folder2}")
        else:
            print(f"No image number found in filename {filename1}")

    return psnr_values

def apply_transformation(test_dir, output_dir, transform):
    """
    Apply transformation to each image in the test directory and save the transformed image in the output directory.

    Args:
    - test_dir (str): Path to the directory containing the images to be transformed.
    - output_dir (str): Path to the directory where the transformed images will be saved.
    - transform (callable): Transformation function to apply to the images.
    """
    # Iterate over each image in the test directory
    for filename in os.listdir(test_dir):
        # Load the image
        image_path = os.path.join(test_dir, filename)
        image = cv2.imread(image_path)

        # Apply the transformation
        transformed_image = transform(image)

        # Save the transformed image
        transformed_image_path = os.path.join(output_dir, f"{filename}")
        cv2.imwrite(transformed_image_path, transformed_image)


def transform(image):
    return cv2.resize(image, (224, 224))


if __name__ == "__main__":


    folder1 = "..\\image-deblurring-using-deep-learning\\test_data\\sharp_gaussian_resized"
    folder2 = "..\\image-deblurring-using-deep-learning\\test_data\\test_deblurred_images_final_results"

    psnr_values = calculate_folder_psnr(folder1, folder2)
    #average_psnr = sum(psnr for _, psnr in psnr_values) / len(psnr_values)
    #print(f"Average PSNR: {average_psnr:.2f} dB")

    if psnr_values:
        average_psnr = sum(psnr for psnr in psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr:.2f} dB")
    else:
        print("No PSNR values calculated. Check input folders.")

'''
# resizing images to calculate psnr between sharp and deblurred 

test_dir = "..\\image-deblurring-using-deep-learning\\test_data\\sharp"
output_dir = "..\\image-deblurring-using-deep-learning\\test_data\\sharp_gaussian_resized"

apply_transformation(test_dir, output_dir, transform)
'''
