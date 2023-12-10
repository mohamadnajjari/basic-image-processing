
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt

def load_and_display_original_image():
    original_image = cv2.imread('party.jpg')
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(original_image_rgb)
    ax.set_title('Original Image')
    plt.show()
    return original_image_rgb

def apply_image_processing_operations(original_image_rgb):
    image_darken = np.clip(original_image_rgb.astype(np.int16) - 128, 0, 255).astype(np.uint8)
    image_lighten = np.clip(original_image_rgb.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    image_invert = np.clip(255 - original_image_rgb.astype(np.int16), 0, 255).astype(np.uint8)
    image_low_contrast = np.clip(original_image_rgb.astype(np.int16) / 2, 0, 255).astype(np.uint8)
    image_high_contrast = np.clip(original_image_rgb.astype(np.int16) * 2, 0, 255).astype(np.uint8)
    image_grayscale = np.clip(0.3 * original_image_rgb[:, :, 0] + 0.6 * original_image_rgb[:, :, 1] + 0.1 * original_image_rgb[:, :, 2], 0, 255)

    return image_darken, image_lighten, image_invert, image_low_contrast, image_high_contrast, image_grayscale

def display_processed_images(processed_images):
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    titles = ['Darken Image', 'Lighten Image', 'Invert Image', 'Low Contrast Image', 'High Contrast Image', 'Grayscale Image']

    for i in range(2):
        for j in range(3):
            index = i * 3 + j
            cmap = 'gray' if titles[index] == 'Grayscale Image' else None
            ax[i, j].imshow(processed_images[index], cmap=cmap)
            ax[i, j].set_title(titles[index])
    
    plt.show()

original_image_rgb = load_and_display_original_image()
processed_images = apply_image_processing_operations(original_image_rgb)
display_processed_images(processed_images)