import os
import cv2
import numpy as np
import requests
import zipfile
import glob
from pycocotools.coco import COCO
from PIL import Image, ImageFilter
import random
import shutil
import random

def download_annotations(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # get annotations file
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

    print("Annotations file downloaded successfully.")


def extract_annotations(zip_path, extract_path):
    # extract annotations from zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Annotations extracted successfully.")


def download_images(coco, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # get all image IDs
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        image_url = image_info['coco_url']
        image_path = os.path.join(save_dir, image_info['file_name'])

        # download image
        response = requests.get(image_url)
        with open(image_path, 'wb') as file:
            file.write(response.content)

        print("Downloaded image:", image_path)


def construct_output_path(image_path, output_dir, prefix="blurred_"):
    # extracting directory and filename from image_path
    directory, filename = os.path.split(image_path)

    # constructing the output directory path
    output_directory = os.path.join(directory, output_dir)
    os.makedirs(output_directory, exist_ok=True)

    # constructing the output path for the blurred image
    output_path = os.path.join(output_directory, f"{prefix}{filename}")

    return output_path


def simple_blur(image_path, output_dir="blurred_images_simple"):
    img = Image.open(image_path)
    blurred_img = img.filter(ImageFilter.BLUR)

    output_path = construct_output_path(image_path, output_dir, prefix="simple_blur_")
    # saving blur image
    blurred_img.save(output_path)
    #print("Blurred image saved at:", output_path)


def box_blur(image_path, radius=2, output_dir="blurred_images_box"):
    img = Image.open(image_path)
    blurred_img = img.filter(ImageFilter.BoxBlur(radius))

    output_path = construct_output_path(image_path, output_dir, prefix="box_blur_")
    # saving blur image
    blurred_img.save(output_path)
    #print("Blurred image saved at:", output_path)


def gaussian_blur(image_path, radius=2, output_dir="blurred_images_gaussian"):
    img = Image.open(image_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))

    output_path = construct_output_path(image_path, output_dir, prefix="gauss_blur_")
    # saving blur image
    blurred_img.save(output_path)
    #print("Blurred image saved at:", output_path)

# take from https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
def motion_blur(image_path, output_dir="blurred_images_motion"):
    img = cv2.imread(image_path)

    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 30

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)

    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    # Combine the vertical and horizontal motion blur images
    blurred_img = cv2.addWeighted(vertical_mb, 0.5, horizonal_mb, 0.5, 0)

    # Construct the output path
    output_path = construct_output_path(image_path, output_dir, prefix="motion_blur_")

    # Save the blurred image
    cv2.imwrite(output_path, blurred_img)

    #print("Blurred image saved at:", output_path)


def test_blur_funcs():
    ###### TESTING BLURRING #########
    input_image_path = "./coco_images/COCO_train2014_000000000625.jpg"
    # simple blur
    print("Applying simple blur...")
    simple_blur(input_image_path)
    print()

    # box blur
    print("Applying box blur...")
    box_blur(input_image_path, radius=2)
    print()

    #  Gaussian blur
    print("Applying Gaussian blur...")
    gaussian_blur(input_image_path, radius=2)
    print()

    #  motion blur
    print("Applying motion blur...")
    motion_blur(input_image_path)
    print()


def split_dataset(root_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # create folders for train, val, and test sets
    train_folder = os.path.join(root_folder, 'train')
    val_folder = os.path.join(root_folder, 'val')
    test_folder = os.path.join(root_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # listing all image files in the root folder
    image_files = [f for f in os.listdir(root_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the number of images for each set
    num_images = len(image_files)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)
    num_test = num_images - num_train - num_val

    # assigning images to train, val, and test sets
    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    # relocate images to their respective folders
    move_images(train_images, root_folder, train_folder)
    move_images(val_images, root_folder, val_folder)
    move_images(test_images, root_folder, test_folder)


def move_images(images, source_folder, destination_folder):
    for image in images:
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, image)
        shutil.move(source_path, destination_path)

def get_images(annotations_url, annotations_save_path, images_save_dir):
    # download annotations
    download_annotations(annotations_url, annotations_save_path)

    # Eextract annotations
    extract_annotations(annotations_save_path, './annotations')

    # init coco api
    coco = COCO('./annotations/annotations/instances_train2014.json')

    # download images
    download_images(coco, images_save_dir)

def main():
    #### GETTING IMAGES ######
    # coco annotations URL
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    # path to save annotations
    annotations_save_path = './annotations_trainval2014.zip'
    # path to images
    images_save_dir = './coco_images/'

    # get_images(annotations_url, annotations_save_path, images_save_dir)

    #### TEST THE BLURRING FUNCTIONS ####
    # test_blur_funcs()

    #### CREATE BLURRED IMAGES ####
    image_paths = glob.glob(r".\coco_images\*.jpg")  # getting a list of all .jpeg files

    # shuffle and split the images into train, test and val-groups
    random.shuffle(image_paths)
    train_size = 0.7
    test_size = 0.15
    val_size = 1-train_size-test_size
    n = len(image_paths)

    train_paths = image_paths[:int(n*train_size)]
    test_paths = image_paths[int(n*train_size):int(n*train_size)+int(n*test_size)]
    val_paths = image_paths[int(n*train_size)+int(n*test_size):]

    print("starting train")
    for image in train_paths:
        simple_blur(image, output_dir=r"train_test_val-splits\simple\train")
        box_blur(image, output_dir=r"train_test_val-splits\box\train")
        gaussian_blur(image, output_dir=r"train_test_val-splits\gaussian\train")
        motion_blur(image, output_dir=r"train_test_val-splits\motion\train")
        img = Image.open(image)
        output_path = construct_output_path(image, output_dir=r"train_test_val-splits\sharp\train", prefix="sharp_")
        img.save(output_path)
    print("starting test")
    for image in test_paths:
        simple_blur(image, output_dir=r"train_test_val-splits\simple\test")
        box_blur(image, output_dir=r"train_test_val-splits\box\test")
        gaussian_blur(image, output_dir=r"train_test_val-splits\gaussian\test")
        motion_blur(image, output_dir=r"train_test_val-splits\motion\test")
        img = Image.open(image)
        output_path = construct_output_path(image, output_dir=r"train_test_val-splits\sharp\test", prefix="sharp_")
        img.save(output_path)
    print("starting val")
    for image in val_paths:
        simple_blur(image, output_dir=r"train_test_val-splits\simple\val")
        box_blur(image, output_dir=r"train_test_val-splits\box\val")
        gaussian_blur(image, output_dir=r"train_test_val-splits\gaussian\val")
        motion_blur(image, output_dir=r"train_test_val-splits\motion\val")
        img = Image.open(image)
        output_path = construct_output_path(image, output_dir=r"train_test_val-splits\sharp\val", prefix="sharp_")
        img.save(output_path)


    """
    image_paths = glob.glob(r".\coco_images\*.jpg")     # getting a list of all .jpeg files

    for image in image_paths:
        simple_blur(image)
        box_blur(image)
        gaussian_blur(image)
        motion_blur(image)
 """
    root_folder = './coco_images/blurred_images_gaussian'
    split_dataset(root_folder)

if __name__ == "__main__":
    main()