# batch_processing.py

import os
import cv2
import logging
from pathlib import Path
from models.face_detection import FaceDetector
from models.face_enhancement import FaceEnhancer
from models.face_recognition import FaceRecognizer
from models.database_manager import DatabaseManager
from utils.image_utils import ImageUtils
from utils.preprocessing import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BatchProcessor:
    """
    Batch process a directory of images (or videos) for face recognition.
    """

    def __init__(self, detector_method='deepface'):
        self.detector = FaceDetector(method=detector_method)
        self.enhancer = FaceEnhancer()
        self.recognizer = FaceRecognizer()
        self.db = DatabaseManager()
        self.preprocessor = Preprocessor(resize=(224, 224))

    def process_directory(self, input_dir, output_dir):
        """
        Process all images in input_dir, annotate, and save to output_dir.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            for img_path in input_dir.glob(ext):
                try:
                    logger.info(f"Processing {img_path.name}")
                    img = ImageUtils.load_image(str(img_path))
                    if img is None:
                        continue

                    faces = self.detector.detect_faces(img)
                    for box in faces:
                        # align and crop
                        face = self.preprocessor.align_and_crop(img, box)
                        if face is None:
                            continue

                        # enhance low-quality
                        if face.shape[0] < 100:
                            face = self.enhancer.enhance_low_quality_image(face, method='combined')

                        # optional augmentation (comment out if not needed)
                        # face = self.preprocessor.augment(face)

                        # generate encoding and recognize
                        encoding = self.recognizer.generate_encoding(face)
                        name, distance = self.db.find_best_match(encoding, self.recognizer)
                        confidence = 1.0 - distance if name != "Unknown" else 0.0

                        # annotate
                        annotated = ImageUtils.draw_face_box(img, box, name, confidence)
                    
                    # save annotated image
                    save_path = output_dir / img_path.name
                    ImageUtils.save_image(annotated, str(save_path))

                except Exception as e:
                    logger.error(f"Failed to process {img_path.name}: {e}")
