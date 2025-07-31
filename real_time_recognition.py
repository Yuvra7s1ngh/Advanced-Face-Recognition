import cv2
import numpy as np
import time
from collections import deque
import logging

from models.face_recognition import FaceRecognizer
from models.face_detection import FaceDetector
from models.face_enhancement import FaceEnhancer
from models.database_manager import DatabaseManager
from utils.video_utils import VideoUtils
from utils.image_utils import ImageUtils
from config.config import VIDEO_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeFaceRecognition:
    def __init__(self):
        """Initialize real-time face recognition system"""
        self.face_recognizer = FaceRecognizer()
        self.face_detector = FaceDetector(method='deepface')
        self.face_enhancer = FaceEnhancer()
        self.database_manager = DatabaseManager()
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.recognition_history = deque(maxlen=10)  # For smoothing results
        
        # Status
        self.is_running = False
        
        logger.info("Real-time face recognition system initialized")
        logger.info(f"Loaded {len(self.database_manager.face_names)} known faces")
    
    def start_recognition(self, source=0, display=True, save_output=False, output_path=None):
        """
        Start real-time face recognition
        source: webcam index, video file path, or RTSP URL
        display: whether to display the video feed
        save_output: whether to save processed video
        output_path: path to save output video
        """
        try:
            # Open video source
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            
            cap = VideoUtils.open_video_source(source)
            if cap is None:
                logger.error("Failed to open video source")
                return False
            
            # Get video info
            video_info = VideoUtils.get_video_info(cap)
            logger.info(f"Video info: {video_info}")
            
            # Setup video writer if saving output
            writer = None
            if save_output and output_path:
                writer = VideoUtils.create_video_writer(
                    output_path, 
                    video_info['fps'],
                    video_info['width'],
                    video_info['height']
                )
            
            self.is_running = True
            frame_count = 0
            last_fps_time = time.time()
            
            logger.info("Starting real-time recognition... Press 'q' to quit")
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % VIDEO_CONFIG['frame_skip'] != 0:
                    continue
                
                start_time = time.time()
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Calculate FPS
                process_time = time.time() - start_time
                fps = 1.0 / max(process_time, 0.001)
                self.fps_queue.append(fps)
                
                # Add FPS to frame
                avg_fps = np.mean(self.fps_queue)
                cv2.putText(processed_frame, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                if display:
                    cv2.imshow('Real-Time Face Recognition', processed_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received")
                        break
                    elif key == ord('s') and not save_output:
                        # Save current frame
                        timestamp = int(time.time())
                        save_path = f"captured_frame_{timestamp}.jpg"
                        ImageUtils.save_image(processed_frame, save_path)
                        logger.info(f"Saved frame to {save_path}")
                
                # Save to output video
                if writer:
                    writer.write(processed_frame)
                
                # Print periodic status
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames, Average FPS: {avg_fps:.1f}")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            self.is_running = False
            logger.info(f"Recognition session completed. Processed {frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error in real-time recognition: {e}")
            return False
    
    def _process_frame(self, frame):
        """Process a single frame for face recognition"""
        try:
            # Resize frame for better performance
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return display_frame
            
            # Process each detected face
            for face_box in faces:
                try:
                    # Extract face region
                    face_region = self.face_detector.extract_face_region(frame, face_box)
                    
                    if face_region.size == 0:
                        continue
                    
                    # Enhance if low quality
                    if face_region.shape[0] < 80 or face_region.shape[1] < 80:
                        face_region = self.face_enhancer.enhance_low_quality_image(
                            face_region, method='combined'
                        )
                    
                    # Generate encoding
                    encoding = self.face_recognizer.generate_encoding(face_region)
                    
                    if encoding is not None:
                        # Find best match in database
                        name, distance = self.database_manager.find_best_match(
                            encoding, self.face_recognizer
                        )
                        
                        # Apply smoothing to reduce flickering
                        name = self._smooth_recognition_result(name)
                        
                        # Determine color based on recognition result
                        if name != "Unknown":
                            color = (0, 255, 0)  # Green for known faces
                            confidence = 1.0 - distance
                        else:
                            color = (0, 0, 255)  # Red for unknown faces
                            confidence = 0.0
                        
                        # Draw face box and label
                        display_frame = ImageUtils.draw_face_box(
                            display_frame, face_box, name, confidence, color
                        )
                    else:
                        # Could not generate encoding
                        display_frame = ImageUtils.draw_face_box(
                            display_frame, face_box, "No Encoding", 0.0, (255, 0, 0)
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue
            
            return display_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def _smooth_recognition_result(self, current_name):
        """Smooth recognition results to reduce flickering"""
        self.recognition_history.append(current_name)
        
        # Count occurrences of each name
        name_counts = {}
        for name in self.recognition_history:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        # Return most frequent name
        if name_counts:
            return max(name_counts, key=name_counts.get)
        else:
            return current_name
    
    def stop_recognition(self):
        """Stop the recognition process"""
        self.is_running = False
        logger.info("Stopping real-time recognition...")

def main():
    """Main function for standalone usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Face Recognition")
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (webcam index, file path, or RTSP URL)')
    parser.add_argument('--no-display', action='store_true', 
                       help='Run without displaying video feed')
    parser.add_argument('--save', type=str, 
                       help='Save output video to specified path')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit
    source = args.source
    if source.isdigit():
        source = int(source)
    
    recognizer = RealTimeFaceRecognition()
    recognizer.start_recognition(
        source=source,
        display=not args.no_display,
        save_output=bool(args.save),
        output_path=args.save
    )

if __name__ == "__main__":
    main()
