import cv2
import numpy as np
import albumentations as A
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Image preprocessing utilities for face recognition.
    """

    def __init__(self, resize=(224, 224)):
        self.resize = resize
        # Define a standard augmentation pipeline (optional use)
        self.aug_pipeline = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.5),
            A.HueSaturationValue(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize()
        ])

    def align_and_crop(self, image, face_box, padding=10):
        """
        Align (crop with padding) and resize a face region.
        :param image: RGB image as numpy array
        :param face_box: [x, y, w, h]
        :param padding: int pixels of padding around box
        :return: preprocessed face region
        """
        x, y, w, h = face_box
        h_img, w_img = image.shape[:2]

        # compute padded coordinates
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            logger.warning("Empty face region after crop")
            return None

        face_resized = cv2.resize(face, self.resize, interpolation=cv2.INTER_CUBIC)
        return face_resized

    def normalize(self, image):
        """
        Normalize pixel values to range [0,1] and standardize to zero-mean unit-variance.
        :param image: numpy array, uint8
        :return: numpy array, float32
        """
        img = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def augment(self, image):
        """
        Apply random augmentations.
        :param image: numpy array, uint8
        :return: numpy array, float32 normalized
        """
        try:
            augmented = self.aug_pipeline(image=image)
            return augmented['image']
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            # Fallback to basic normalize
            return self.normalize(image)
