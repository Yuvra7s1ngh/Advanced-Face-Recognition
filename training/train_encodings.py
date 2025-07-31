import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from models.face_recognition import FaceRecognizer
from models.face_detection import FaceDetector
from models.face_enhancement import FaceEnhancer
from models.database_manager import DatabaseManager
from utils.image_utils import ImageUtils
from config.config import KNOWN_FACES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEncodingTrainer:
    def __init__(self):
        """Initialize the face encoding trainer"""
        self.face_recognizer = FaceRecognizer()
        self.face_detector = FaceDetector(method='deepface')
        self.face_enhancer = FaceEnhancer()
        self.database_manager = DatabaseManager()

    def train_from_directory(self, directory_path=None):
        """
        Train face encodings from directory structure
        Expected structure: directory_path/person_name/image_files
        """
        if directory_path is None:
            directory_path = KNOWN_FACES_DIR

        directory_path = Path(directory_path)

        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return False

        logger.info(f"Training face encodings from: {directory_path}")

        total_processed = 0
        total_failed = 0

        # Process each person's directory
        for person_dir in directory_path.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            logger.info(f"Processing images for: {person_name}")

            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
                image_files.extend(person_dir.glob(ext.upper()))

            if not image_files:
                logger.warning(f"No image files found for {person_name}")
                continue

            person_processed = 0
            person_failed = 0

            # Process each image
            for image_path in tqdm(image_files, desc=f"Processing {person_name}"):
                try:
                    success = self._process_single_image(str(image_path), person_name)
                    if success:
                        person_processed += 1
                        total_processed += 1
                    else:
                        person_failed += 1
                        total_failed += 1

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    person_failed += 1
                    total_failed += 1

            logger.info(f"Completed {person_name}: {person_processed} successful, {person_failed} failed")

        # Save encodings to file
        if total_processed > 0:
            success = self.database_manager.save_encodings()
            if success:
                logger.info(f"Training completed! Total: {total_processed} successful, {total_failed} failed")
                return True
            else:
                logger.error("Failed to save encodings")
                return False
        else:
            logger.warning("No encodings were generated")
            return False

    def _process_single_image(self, image_path, person_name):
        """Process a single image and add encoding to database"""
        try:
            # Load image
            image = ImageUtils.load_image(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False

            # Detect faces in image
            faces = self.face_detector.detect_faces(image)
            if not faces:
                logger.warning(f"No faces detected in: {image_path}")
                return False

            # Use the largest face if multiple detected
            if len(faces) > 1:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                logger.info(f"Multiple faces detected, using largest in: {image_path}")
            else:
                largest_face = faces[0]

            # Extract face region
            face_region = self.face_detector.extract_face_region(image, largest_face)

            # Enhance face if it's low quality
            if face_region.shape[0] < 100 or face_region.shape[1] < 100:
                face_region = self.face_enhancer.enhance_low_quality_image(
                    face_region, method='combined'
                )

            # Generate encoding
            encoding = self.face_recognizer.generate_encoding(face_region)
            if encoding is None:
                logger.warning(f"Could not generate encoding for: {image_path}")
                return False

            # Add to database
            success = self.database_manager.add_face(person_name, encoding, image_path)
            if success:
                logger.debug(f"Added encoding for {person_name} from {image_path}")
                return True
            else:
                logger.warning(f"Failed to add encoding for {person_name}")
                return False

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return False

    def add_single_face(self, image_path, person_name):
        """Add a single face to the database"""
        try:
            success = self._process_single_image(image_path, person_name)
            if success:
                # Save immediately for single additions
                self.database_manager.save_encodings()
                logger.info(f"Successfully added face for {person_name}")
                return True
            else:
                logger.error(f"Failed to add face for {person_name}")
                return False
        except Exception as e:
            logger.error(f"Error adding single face: {e}")
            return False

    def retrain_person(self, person_name):
        """Retrain encodings for a specific person"""
        try:
            # Remove existing encodings for this person
            self.database_manager.remove_face(person_name)

            # Retrain from their directory
            person_dir = KNOWN_FACES_DIR / person_name
            if not person_dir.exists():
                logger.error(f"Directory for {person_name} does not exist")
                return False

            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
                image_files.extend(person_dir.glob(ext.upper()))

            if not image_files:
                logger.error(f"No images found for {person_name}")
                return False

            processed = 0
            for image_path in image_files:
                success = self._process_single_image(str(image_path), person_name)
                if success:
                    processed += 1

            if processed > 0:
                self.database_manager.save_encodings()
                logger.info(f"Retrained {person_name} with {processed} images")
                return True
            else:
                logger.error(f"Failed to retrain {person_name}")
                return False

        except Exception as e:
            logger.error(f"Error retraining {person_name}: {e}")
            return False

def main():
    """Main training function"""
    trainer = FaceEncodingTrainer()

    # Train from the known faces directory
    success = trainer.train_from_directory()

    if success:
        print("Training completed successfully!")

        # Print database info
        db_info = trainer.database_manager.export_database_info()
        print(f"Database contains {db_info['total_people']} people with {db_info['total_encodings']} total encodings")

        for name, info in db_info['people'].items():
            print(f"  {name}: {info['encoding_count']} encodings")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()