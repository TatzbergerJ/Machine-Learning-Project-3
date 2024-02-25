import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import models
import torch

from torchvision.transforms import transforms
from torchvision.utils import save_image

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

device = 'cpu'

# load the trained model
model = models.CNN().to(device).eval()
model.load_state_dict(torch.load('../outputs/model.pth'))

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# our testing of the model
# Path to the directory containing test images
test_dir = "../test_data/gaussian_blurred/"

# Output directory to save deblurred images
output_dir = "../test_data/test_deblurred_images_final_results/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image in the test directory
for filename in os.listdir(test_dir):
    # Load the image
    image_path = os.path.join(test_dir, filename)
    image = cv2.imread(image_path)
    orig_image = image.copy()
    orig_image = cv2.resize(orig_image, (224, 224))
    cv2.imwrite(os.path.join(output_dir, f"original_blurred_{filename}"), orig_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply transforms and prepare input for the model
    image = transform(image).unsqueeze(0)

    # Apply the model
    with torch.no_grad():
        outputs = model(image)

    # Save the deblurred image
    save_decoded_image(outputs.cpu().data, name=os.path.join(output_dir, f"{filename}"))

'''
name = 'image_1'

image = cv2.imread(f"../test_data/gaussian_blurred/{name}.jpg")
orig_image = image.copy()
orig_image = cv2.resize(orig_image, (224, 224))
cv2.imwrite(f"../outputs/test_deblurred_images/original_blurred_image_2.jpg", orig_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = transform(image).unsqueeze(0)
print(image.shape)

with torch.no_grad():
    outputs = model(image)
    save_decoded_image(outputs.cpu().data, name=f"../outputs/test_deblurred_images/deblurred_image_2.jpg")

'''