import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
ENCODINGS_DIR = DATA_DIR / "encodings"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# Ensure directories exist
for directory in [DATA_DIR, KNOWN_FACES_DIR, ENCODINGS_DIR, TEST_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# DeepFace Configuration
DEEPFACE_CONFIG = {
    'model_name': 'Facenet512',  # High accuracy model
    'detector_backend': 'retinaface',  # Best detector for accuracy
    'distance_metric': 'cosine',
    'enforce_detection': True,
    'align': True,
    'normalization': 'base'
}

# Face Detection Configuration
DETECTION_CONFIG = {
    'min_face_size': 40,
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'confidence_threshold': 0.8
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'frame_skip': 2,  # Process every 2nd frame for performance
    'max_fps': 30,
    'resize_width': 640,
    'resize_height': 480
}

# Recognition Thresholds
RECOGNITION_THRESHOLDS = {
    'Facenet512': 0.4,
    'VGG-Face': 0.6,
    'ArcFace': 0.68,
    'OpenFace': 0.1
}

# Database Configuration
DATABASE_CONFIG = {
    'encodings_file': str(ENCODINGS_DIR / 'face_encodings.pkl'),
    'backup_encodings': True,
    'max_images_per_person': 10
}

# CCTV Configuration
CCTV_CONFIG = {
    'rtsp_timeout': 10,
    'reconnect_attempts': 3,
    'buffer_size': 1
}
