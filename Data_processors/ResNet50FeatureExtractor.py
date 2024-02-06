from AbstractComponents import AbstractImageProcessor
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


class ResNet50FeatureExtractor(AbstractImageProcessor):
    def __init__(self, raw_data_path, extracted_features_path):
        self.raw_data_path = raw_data_path
        self.extracted_features_path = extracted_features_path
        teaching_images, image_features, image_labels = self.perform_feature_extraction()
        super().__init__(raw_data_path, extracted_features_path, teaching_images, image_features, image_labels)

    @staticmethod
    def _resize_image_to_resnet_expected_shape(image):
        # Resize image to (224, 224)
        resized_image = tf.image.resize(image, (224, 224))
        # Convert to 3 channels if grayscale
        if resized_image.shape[-1] == 1:
            resized_image = tf.image.grayscale_to_rgb(resized_image)
        # Rescale pixel values to [0, 1]
        resized_image /= 255.0
        return resized_image

    @staticmethod
    def _get_image_label(image_path):
        # Extract label from image filename or path
        # Assuming filename format is index_label.extension e.g., "4_1.png" is the fourth image with label 1
        label = os.path.basename(image_path).split('.')[0][-1]
        return label

    @staticmethod
    def _is_valid_resnet_input(image):
        return image.shape == (224, 224, 3)

    def _load_images(self):
        image_paths = [os.path.join(self.raw_data_path, img) for img in os.listdir(self.raw_data_path)]
        return image_paths

    def _extract_features(self, image):
        image = img_to_array(image)
        if not self._is_valid_resnet_input(image):
            image = self._resize_image_to_resnet_expected_shape(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(np.copy(image))
        model = ResNet50(weights='imagenet', include_top=False)
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-2].output  # Output of the last convolutional layer
        )
        features = feature_extractor.predict(image)
        return features.flatten()

    def perform_feature_extraction(self):
        # first check if features exist - if so, simply load them
        if os.path.exists(os.path.join(self.extracted_features_path, 'images.npy')):
            print("Features have previously been extracted and saved. Loading Features...")
            return np.load(os.path.join(self.extracted_features_path, 'images.npy')), np.load(os.path.join(self.extracted_features_path, 'features.npy')), np.load(
                os.path.join(self.extracted_features_path, 'labels.npy'))
        else:
            print("Processing images and saving features...")
            images, image_features, image_labels = [], [], []
            image_paths = self._load_images()
            for path in image_paths:
                image = Image.open(path)
                images.append(path)
                features = self._extract_features(image)
                label = self._get_image_label(path)
                image_features.append(features)
                image_labels.append(label)
            self._save_features_and_labels(images, image_features, image_labels)
            return images, image_features, image_labels

    def _save_features_and_labels(self, images, image_features, image_labels):
        os.makedirs(self.extracted_features_path, exist_ok=True)
        np.save(os.path.join(self.extracted_features_path, 'images.npy'), np.array(images))
        np.save(os.path.join(self.extracted_features_path, 'features.npy'), np.array(image_features))
        np.save(os.path.join(self.extracted_features_path, 'labels.npy'), np.array(image_labels))



