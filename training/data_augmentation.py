# training/data_augmentation.py

import os
import cv2
import numpy as np
import albumentations as A
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataAugmentor:
    """
    Generate augmented images to enrich face datasets.
    """

    def __init__(self, output_size=(224,224)):
        self.output_size = output_size
        # Define augmentation pipeline
        self.pipeline = A.Compose([
            A.RandomBrightnessContrast(p=0.7),
            A.GaussianNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Normalize()
        ])

    def augment_person(self, person_dir, augment_count=50):
        """
        Generate augment_count augmented images for each original image in person_dir.
        :param person_dir: Path to folder containing original face images.
        :param augment_count: number of augmentations per original image.
        """
        person_dir = Path(person_dir)
        augmented_dir = person_dir / "augmented"
        augmented_dir.mkdir(exist_ok=True)

        image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        if not image_files:
            logger.warning(f"No images found in {person_dir}")
            return

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read {img_path}")
                continue

            base_name = img_path.stem
            for i in range(augment_count):
                try:
                    augmented = self.pipeline(image=img)['image']
                    # Convert normalized to uint8
                    aug = ((augmented - augmented.min()) / (augmented.max()-augmented.min()) * 255).astype(np.uint8)
                    aug_resized = cv2.resize(aug, self.output_size, interpolation=cv2.INTER_CUBIC)
                    save_name = f"{base_name}_aug_{i+1}.jpg"
                    cv2.imwrite(str(augmented_dir / save_name), aug_resized)
                except Exception as e:
                    logger.error(f"Augmentation error for {img_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data augmentation for face datasets")
    parser.add_argument('--input', type=str, required=True,
                        help="Directory of person folders (each contains images)")
    parser.add_argument('--count', type=int, default=50,
                        help="Augmentations per image")
    args = parser.parse_args()

    input_dir = Path(args.input)
    for person_folder in input_dir.iterdir():
        if person_folder.is_dir():
            logger.info(f"Augmenting data for {person_folder.name}")
            augmentor = DataAugmentor()
            augmentor.augment_person(person_folder, augment_count=args.count)

if __name__ == "__main__":
    main()
