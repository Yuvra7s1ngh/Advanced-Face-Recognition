import cv2
import numpy as np
from config.config import VIDEO_CONFIG
import logging

logger = logging.getLogger(__name__)

class VideoUtils:
    @staticmethod
    def open_video_source(source):
        """
        Open video source (webcam, file, RTSP, or IP camera stream).
        source: int for webcam, str for file path or stream URL
        """
        try:
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                logger.error(f"Could not open video source: {source}")
                return None
            
            # Set buffer size for RTSP streams
            if isinstance(source, str) and source.startswith('rtsp'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, VIDEO_CONFIG.get('buffer_size', 1))
            
            # Set frame dimensions if specified in config
            if VIDEO_CONFIG.get('resize_width') and VIDEO_CONFIG.get('resize_height'):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_CONFIG['resize_width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_CONFIG['resize_height'])
            
            logger.info(f"Opened video source: {source}")
            return cap
            
        except Exception as e:
            logger.error(f"Error opening video source {source}: {e}")
            return None

    @staticmethod
    def get_video_info(cap):
        """Get video capture information"""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            return {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None

    @staticmethod
    def process_video_file(video_path, frame_processor, output_path=None):
        """Process video file or stream frame by frame"""
        try:
            cap = VideoUtils.open_video_source(video_path)
            if cap is None:
                return False
            
            # Get video info
            video_info = VideoUtils.get_video_info(cap)
            if not video_info:
                logger.warning("Could not get video info, using default values.")
                video_info = {'fps': 25, 'width': 640, 'height': 480, 'total_frames': 0}
            
            # Setup video writer if output path provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, video_info['fps'],
                                       (video_info['width'], video_info['height']))
            
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if VIDEO_CONFIG.get('frame_skip', 1) > 1 and frame_count % VIDEO_CONFIG['frame_skip'] != 0:
                    continue
                
                # Process frame
                processed_frame = frame_processor(frame)
                processed_frames += 1
                
                # Write to output if specified
                if writer is not None:
                    writer.write(processed_frame)
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            
            logger.info(f"Processed {processed_frames} frames from {frame_count} total")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False

    @staticmethod
    def create_video_writer(output_path, fps, width, height):
        """Create video writer object"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            return writer
        except Exception as e:
            logger.error(f"Error creating video writer: {e}")
            return None