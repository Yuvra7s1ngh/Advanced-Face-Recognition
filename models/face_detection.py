import cv2
import numpy as np
import dlib
from deepface import DeepFace
from config.config import DETECTION_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, method='deepface'):
        """
        Initialize face detector with specified method
        Methods: 'opencv', 'dlib', 'deepface', 'mtcnn'
        """
        self.method = method
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the specified detector"""
        try:
            if self.method == 'opencv':
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            elif self.method == 'dlib':
                self.detector = dlib.get_frontal_face_detector()
            elif self.method in ['deepface', 'mtcnn']:
                # DeepFace handles detection internally
                pass
            else:
                logger.warning(f"Unknown method '{self.method}', falling back to OpenCV.")
                self.method = 'opencv'
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            logger.info(f"Initialized {self.method} face detector")
        except Exception as e:
            logger.error(f"Error initializing detector: {e}")
            self.method = 'opencv'
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def detect_faces(self, image):
        """
        Detect faces in the image.
        Returns a list of bounding boxes: [(x, y, w, h), ...]
        """
        if self.method == 'opencv':
            return self._detect_opencv(image)
        elif self.method == 'dlib':
            return self._detect_dlib(image)
        elif self.method in ['deepface', 'mtcnn']:
            return self._detect_deepface(image)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return []

    def _detect_opencv(self, image):
        """Detect faces using OpenCV Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=DETECTION_CONFIG.get('scale_factor', 1.1),
            minNeighbors=DETECTION_CONFIG.get('min_neighbors', 5),
            minSize=(DETECTION_CONFIG.get('min_face_size', 30), DETECTION_CONFIG.get('min_face_size', 30))
        )
        return faces.tolist() if isinstance(faces, np.ndarray) else []

    def _detect_dlib(self, image):
        """Detect faces using dlib"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face_boxes = []
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_boxes.append([x, y, w, h])
        return face_boxes

    def _detect_deepface(self, image):
        """
        Detect faces using DeepFace (RetinaFace or MTCNN).
        Returns bounding boxes in (x, y, w, h) format.
        """
        try:
            # DeepFace expects a file path or numpy array
            backend = 'retinaface' if self.method == 'deepface' else 'mtcnn'
            # DeepFace.extract_faces returns a list of dicts with 'facial_area'
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                align=False
            )
            face_boxes = []
            for face in faces:
                area = face.get('facial_area')
                if area:
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    face_boxes.append([x, y, w, h])
            return face_boxes
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return []

    def extract_face_region(self, image, face_box, padding=20):
        """Extract face region from image with padding"""
        x, y, w, h = face_box
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        face_region = image[y_start:y_end, x_start:x_end]
        return face_region