import pickle
import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime
from config.config import DATABASE_CONFIG, KNOWN_FACES_DIR, ENCODINGS_DIR
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize database manager"""
        self.encodings_file = DATABASE_CONFIG['encodings_file']
        self.known_faces = {}
        self.face_names = []
        self.face_encodings = []
        self.load_encodings()
    
    def load_encodings(self):
        """Load face encodings from file"""
        try:
            if os.path.exists(self.encodings_file):
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('known_faces', {})
                    self.face_names = data.get('face_names', [])
                    self.face_encodings = data.get('face_encodings', [])
                logger.info(f"Loaded {len(self.face_names)} face encodings")
            else:
                logger.info("No existing encodings file found")
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            self.known_faces = {}
            self.face_names = []
            self.face_encodings = []
    
    def save_encodings(self):
        """Save face encodings to file"""
        try:
            # Create backup if enabled
            if DATABASE_CONFIG['backup_encodings'] and os.path.exists(self.encodings_file):
                backup_file = f"{self.encodings_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.encodings_file, backup_file)
            
            data = {
                'known_faces': self.known_faces,
                'face_names': self.face_names,
                'face_encodings': self.face_encodings,
                'timestamp': datetime.now().isoformat(),
                'total_faces': len(self.face_names)
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.face_names)} face encodings")
            return True
        except Exception as e:
            logger.error(f"Error saving encodings: {e}")
            return False
    
    def add_face(self, name, encoding, image_path=None):
        """Add a face encoding to the database"""
        try:
            if name not in self.known_faces:
                self.known_faces[name] = {
                    'encodings': [],
                    'image_paths': [],
                    'date_added': datetime.now().isoformat()
                }
            
            # Check if we've reached the maximum images per person
            if len(self.known_faces[name]['encodings']) >= DATABASE_CONFIG['max_images_per_person']:
                logger.warning(f"Maximum images per person reached for {name}")
                return False
            
            self.known_faces[name]['encodings'].append(encoding.tolist())
            if image_path:
                self.known_faces[name]['image_paths'].append(image_path)
            
            # Update flat lists for compatibility
            self.face_names.append(name)
            self.face_encodings.append(encoding)
            
            logger.info(f"Added face encoding for {name}")
            return True
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
    
    def remove_face(self, name):
        """Remove all face encodings for a person"""
        try:
            if name in self.known_faces:
                # Remove from known_faces dict
                encodings_count = len(self.known_faces[name]['encodings'])
                del self.known_faces[name]
                
                # Remove from flat lists
                indices_to_remove = [i for i, n in enumerate(self.face_names) if n == name]
                for index in reversed(indices_to_remove):
                    del self.face_names[index]
                    del self.face_encodings[index]
                
                logger.info(f"Removed {encodings_count} encodings for {name}")
                return True
            else:
                logger.warning(f"Face {name} not found in database")
                return False
        except Exception as e:
            logger.error(f"Error removing face: {e}")
            return False
    
    def get_face_names(self):
        """Get list of all known face names"""
        return list(self.known_faces.keys())
    
    def get_face_count(self, name=None):
        """Get count of face encodings"""
        if name:
            return len(self.known_faces.get(name, {}).get('encodings', []))
        else:
            return len(self.face_names)
    
    def find_best_match(self, unknown_encoding, recognizer):
        """Find the best matching face in database"""
        try:
            if len(self.face_encodings) == 0:
                return "Unknown", 1.0
            
            best_match_name = "Unknown"
            best_distance = float('inf')
            
            for i, known_encoding in enumerate(self.face_encodings):
                is_match, distance = recognizer.compare_faces(
                    np.array(known_encoding), unknown_encoding
                )
                
                if distance < best_distance:
                    best_distance = distance
                    if is_match:
                        best_match_name = self.face_names[i]
            
            return best_match_name, best_distance
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return "Unknown", 1.0
    
    def export_database_info(self):
        """Export database information as JSON"""
        try:
            info = {
                'total_people': len(self.known_faces),
                'total_encodings': len(self.face_names),
                'people': {}
            }
            
            for name, data in self.known_faces.items():
                info['people'][name] = {
                    'encoding_count': len(data['encodings']),
                    'date_added': data.get('date_added', 'Unknown'),
                    'image_count': len(data.get('image_paths', []))
                }
            
            return info
        except Exception as e:
            logger.error(f"Error exporting database info: {e}")
            return {}
