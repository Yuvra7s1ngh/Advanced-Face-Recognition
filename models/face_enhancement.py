import cv2
import numpy as np
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)

class FaceEnhancer:
    def __init__(self):
        """Initialize face enhancement utilities"""
        pass
    
    def enhance_low_quality_image(self, image, method='super_resolution'):
        """
        Enhance low quality face image
        Methods: 'super_resolution', 'histogram_eq', 'denoising', 'sharpening', 'combined'
        """
        try:
            if method == 'super_resolution':
                return self._super_resolution_enhancement(image)
            elif method == 'histogram_eq':
                return self._histogram_equalization(image)
            elif method == 'denoising':
                return self._denoise_image(image)
            elif method == 'sharpening':
                return self._sharpen_image(image)
            elif method == 'combined':
                return self._combined_enhancement(image)
            else:
                return image
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            return image
    
    def _super_resolution_enhancement(self, image):
        """Apply super-resolution using OpenCV's DNN super-resolution"""
        try:
            # Simple bicubic upscaling as fallback
            # For production, consider using ESRGAN or Real-ESRGAN
            height, width = image.shape[:2]
            enhanced = cv2.resize(image, (width * 2, height * 2), 
                                interpolation=cv2.INTER_CUBIC)
            return enhanced
        except Exception as e:
            logger.error(f"Super-resolution error: {e}")
            return image
    
    def _histogram_equalization(self, image):
        """Apply histogram equalization to improve contrast"""
        try:
            if len(image.shape) == 3:
                # Convert to YUV and equalize Y channel
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                enhanced = cv2.equalizeHist(image)
            return enhanced
        except Exception as e:
            logger.error(f"Histogram equalization error: {e}")
            return image
    
    def _denoise_image(self, image):
        """Remove noise from image"""
        try:
            if len(image.shape) == 3:
                enhanced = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                enhanced = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            return enhanced
        except Exception as e:
            logger.error(f"Denoising error: {e}")
            return image
    
    def _sharpen_image(self, image):
        """Apply sharpening filter"""
        try:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(image, -1, kernel)
            return enhanced
        except Exception as e:
            logger.error(f"Sharpening error: {e}")
            return image
    
    def _combined_enhancement(self, image):
        """Apply combined enhancement techniques"""
        try:
            # Apply multiple enhancement techniques
            enhanced = self._denoise_image(image)
            enhanced = self._histogram_equalization(enhanced)
            enhanced = self._sharpen_image(enhanced)
            return enhanced
        except Exception as e:
            logger.error(f"Combined enhancement error: {e}")
            return image
    
    def preprocess_for_recognition(self, image):
        """Preprocess image for better recognition accuracy"""
        try:
            # Resize to standard size
            processed = cv2.resize(image, (224, 224))
            
            # Normalize
            processed = processed.astype(np.float32) / 255.0
            
            # Apply slight Gaussian blur to reduce noise
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
            # Convert back to uint8
            processed = (processed * 255).astype(np.uint8)
            
            return processed
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image
