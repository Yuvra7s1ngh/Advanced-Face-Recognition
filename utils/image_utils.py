import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    @staticmethod
    def load_image(image_path):
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image, save_path):
        """Save image to file"""
        try:
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            cv2.imwrite(save_path, image_bgr)
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def resize_image(image, width, height, maintain_aspect=True):
        """Resize image to specified dimensions"""
        try:
            if maintain_aspect:
                h, w = image.shape[:2]
                aspect = w / h
                
                if aspect > width / height:
                    new_w = width
                    new_h = int(width / aspect)
                else:
                    new_h = height
                    new_w = int(height * aspect)
                
                resized = cv2.resize(image, (new_w, new_h))
                
                # Pad to exact dimensions
                delta_w = width - new_w
                delta_h = height - new_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                color = [0, 0, 0]
                resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                           cv2.BORDER_CONSTANT, value=color)
            else:
                resized = cv2.resize(image, (width, height))
            
            return resized
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def draw_face_box(image, face_box, name="Unknown", confidence=0.0, color=(0, 255, 0)):
        """Draw bounding box and label on image"""
        try:
            x, y, w, h = face_box
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            if confidence > 0:
                label = f"{name} ({confidence:.2f})"
            else:
                label = name
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(image, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
        except Exception as e:
            logger.error(f"Error drawing face box: {e}")
            return image
    
    @staticmethod
    def validate_image(image_path):
        """Validate if file is a valid image"""
        try:
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                img.verify()
            
            return True
        except Exception:
            return False
