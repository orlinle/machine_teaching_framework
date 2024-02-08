import numpy as np
from tensorflow.keras.datasets import mnist
import os
from PIL import Image

# Number of images per class to download
NUM_IMAGES_PER_CLASS = 25
def save_images(images, labels, output_dir, image_format='png'):
    os.makedirs(output_dir, exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(output_dir, f"{i}_{label}.{image_format}")
        image = Image.fromarray(image)
        image.save(image_path)


def download_and_save_binary_dataset(num_images_per_class, output_dir, image_format='png'):
    (train_images, train_labels), (_, _) = mnist.load_data()

    selected_images = []
    selected_labels = []
    classes_count = [0, 0]  # Count of images for each class

    for image, label in zip(train_images, train_labels):
        if label == 0:
            if classes_count[0] < num_images_per_class:
                selected_images.append(image)
                selected_labels.append(label)
                classes_count[0] += 1
        elif label == 1:
            if classes_count[1] < num_images_per_class:
                selected_images.append(image)
                selected_labels.append(label)
                classes_count[1] += 1

        if all(count >= num_images_per_class for count in classes_count):
            break

    save_images(selected_images, selected_labels, output_dir, image_format)


# Example usage:
def run_example_usage():
    num_images_per_class = NUM_IMAGES_PER_CLASS  # Number of images per class to download
    output_directory = "Datasets/Binary_mnist"  # Output directory to save images
    download_and_save_binary_dataset(num_images_per_class, output_directory)

# run_example_usage()
