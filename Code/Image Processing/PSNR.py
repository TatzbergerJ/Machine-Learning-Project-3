import cv2
import numpy as np
from math import log10, sqrt


def PSNR(og_img_path: str, db_img_path: str):
    """
    reads in paths of an original image and a deblurred version of the image and returns the PSNR value (in dB) in order
    to compare the pictures
    :param og_img_path: path of the original image
    :param db_img_path: path of the deblurred image
    :return: PSNR ratio value (in dB)
    """
    og_img = cv2.imread(og_img_path)
    db_img = cv2.imread(db_img_path)

    # check if both images are of the same dimensions
    height_og, width_og, _ = og_img.shape
    height_db, width_db, _ = db_img.shape
    ratio_og = height_og/width_og
    ratio_db = height_db/width_db

    # check if the pictures have the same ratio and size, if not: return a message and NaN
    if round(ratio_og, 2) != round(ratio_db, 2):
        print("Images do not have the same ratio, check the input!")
        return float("NaN")

    if (height_og != height_db) or (width_og != width_db):
        print("Images are not of same size, check the input!")
        return float("NaN")

    mse = np.mean((og_img-db_img)**2)
    if mse == 0: # if both pictures are exactly the same
        return 100
    max_pixel = 255.0
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr

if __name__ == "__main__":
    # testing
    image_og = r"C:\Users\tatzb\Desktop\Work,Study\01 - Studium\02_Master (Data Science)\1. Semester\Machine Learning\Projekte\Machine-Learning\03_Deep_Learning\Machine-Learning-Project-3\Code\Image Processing\coco_images\COCO_train2014_000000000625.jpg"
    image_blurred = r"C:\Users\tatzb\Desktop\Work,Study\01 - Studium\02_Master (Data Science)\1. Semester\Machine Learning\Projekte\Machine-Learning\03_Deep_Learning\Machine-Learning-Project-3\Code\Image Processing\coco_images\blurred_images_box\blurred_COCO_train2014_000000000625.jpg"

    test = PSNR(image_og, image_blurred)
    print(test)
