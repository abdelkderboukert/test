import os
import numpy as np
from PIL import Image
import cv2
from scipy.fftpack import dct
from skimage.feature import hog
import pandas as pd

#Bluck

def compute_phog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1)):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=False)
    return hog_features

data_dir = r"C:\Users\HP\rebo\ai-tp\public\Datasets"
output_dir = r"C:\Users\HP\rebo\ai-tp\public"
img_height = 224
img_width = 224
num_classes = 2

# Apply Gabor filter parameters
sigma = 10
theta = np.pi / 2
lambd = np.pi / 4 
phi = 0
gamma = 0.5
kernel = cv2.getGaborKernel((img_height, img_width), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)

images = []
labels = []
image_names = []

i = 0

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            file_extension = os.path.splitext(img_path)[1]
            try:
                myimg = Image.open(img_path).convert('RGB')
                myimg = myimg.resize((img_width, img_height))
                img_array = np.array(myimg)

                img_array = (img_array / 255.0).astype(np.float32)  # Convert to float32 for processing
                
                gabor_image = cv2.filter2D(img_array, cv2.CV_32F, kernel)

                dct_image = dct(dct(gabor_image.T, norm='ortho').T, norm='ortho')

                f_transform = np.fft.fft2(img_array)
                f_transform_shifted = np.fft.fftshift(f_transform)

                phog_features = compute_phog(img_array)

                new_img_file = f"{i}{file_extension}"
                new_img_path = os.path.join(label_dir, new_img_file)
                os.rename(img_path, new_img_path)
                print(f"{new_img_file} is: {label}")

                images.append(gabor_image)
                labels.append(label)
                image_names.append(new_img_file)
                i += 1
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

    

data = {'Image Name': image_names, 'Label': labels, 'image': images}
output_csv_path = os.path.join(output_dir, 'image_labels.csv')
df = pd.DataFrame(data)
df.to_csv('image_labels.csv', index=False)