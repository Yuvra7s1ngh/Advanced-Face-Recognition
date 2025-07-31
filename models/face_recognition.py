import pickle
import numpy as np
from deepface import DeepFace
from config.config import DEEPFACE_CONFIG, RECOGNITION_THRESHOLDS
import logging
import os

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self):
        """Initialize face recognition system"""
        self.model_name = DEEPFACE_CONFIG['model_name']
        self.detector_backend = DEEPFACE_CONFIG['detector_backend']
        self.distance_metric = DEEPFACE_CONFIG['distance_metric']
        self.threshold = RECOGNITION_THRESHOLDS.get(self.model_name, 0.6)
        
        # Build model to ensure it's loaded
        try:
            DeepFace.build_model(self.model_name)
            logger.info(f"Loaded {self.model_name} model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def generate_encoding(self, image):
        """Generate face encoding for an image"""
        try:
            # Generate embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
                # normalization parameter removed for compatibility
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                logger.warning("No face detected for encoding generation")
                return None
                
        except Exception as e:
            logger.error(f"Encoding generation error: {e}")
            return None
    
    def compare_faces(self, known_encoding, unknown_encoding):
        """Compare two face encodings"""
        try:
            if known_encoding is None or unknown_encoding is None:
                return False, 1.0
            
            # Calculate distance based on metric
            if self.distance_metric == 'cosine':
                distance = self._cosine_distance(known_encoding, unknown_encoding)
            elif self.distance_metric == 'euclidean':
                distance = self._euclidean_distance(known_encoding, unknown_encoding)
            elif self.distance_metric == 'euclidean_l2':
                distance = self._euclidean_l2_distance(known_encoding, unknown_encoding)
            else:
                distance = self._cosine_distance(known_encoding, unknown_encoding)
            
            is_match = distance <= self.threshold
            return is_match, distance
            
        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            return False, 1.0
    
    def _cosine_distance(self, encoding1, encoding2):
        """Calculate cosine distance between encodings"""
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance
    
    def _euclidean_distance(self, encoding1, encoding2):
        """Calculate Euclidean distance between encodings"""
        return np.linalg.norm(encoding1 - encoding2)
    
    def _euclidean_l2_distance(self, encoding1, encoding2):
        """Calculate L2 normalized Euclidean distance"""
        encoding1_l2 = encoding1 / np.linalg.norm(encoding1)
        encoding2_l2 = encoding2 / np.linalg.norm(encoding2)
        return np.linalg.norm(encoding1_l2 - encoding2_l2)
    
    def verify_face(self, image1, image2):
        """Verify if two images contain the same person"""
        try:
            result = DeepFace.verify(
                img1_path=image1,
                img2_path=image2,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                align=True,
                normalization='base'
            )
            
            return result['verified'], result['distance']
            
        except Exception as e:
            logger.error(f"Face verification error: {e}")
            return False, 1.0
    
    def find_face_in_database(self, image, db_path):
        """Find matching face in database"""
        try:
            results = DeepFace.find(
                img_path=image,
                db_path=db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True
            )
            
            if len(results) > 0 and len(results[0]) > 0:
                # Return the best match
                best_match = results[0].iloc[0]
                return best_match['identity'], best_match[f'{self.model_name}_{self.distance_metric}']
            
            return None, 1.0
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return None, 1.0
