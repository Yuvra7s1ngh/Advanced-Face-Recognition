import cv2
import numpy as np
import time
import threading
from datetime import datetime
import json
import logging
from pathlib import Path

from models.face_recognition import FaceRecognizer
from models.face_detection import FaceDetector
from models.face_enhancement import FaceEnhancer
from models.database_manager import DatabaseManager
from utils.video_utils import VideoUtils
from utils.image_utils import ImageUtils
from config.config import CCTV_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CCTVProcessor:
    def __init__(self):
        """Initialize CCTV processing system"""
        self.face_recognizer = FaceRecognizer()
        self.face_detector = FaceDetector(method='deepface')
        self.face_enhancer = FaceEnhancer()
        self.database_manager = DatabaseManager()
        
        # CCTV specific settings
        self.streams = {}  # Dictionary to store multiple streams
        self.is_monitoring = False
        self.detection_log = []
        
        # Alert settings
        self.alert_threshold = 0.8  # Confidence threshold for alerts
        self.unknown_person_alerts = True
        self.known_person_alerts = False
        
        # Storage settings
        self.save_detections = True
        self.detection_save_path = Path("detections")
        self.detection_save_path.mkdir(exist_ok=True)
        
        logger.info("CCTV processor initialized")
    
    def add_stream(self, stream_id, rtsp_url, location_name="Unknown"):
        """Add a CCTV stream for monitoring"""
        try:
            cap = VideoUtils.open_video_source(rtsp_url)
            if cap is None:
                logger.error(f"Failed to open RTSP stream: {rtsp_url}")
                return False
            
            self.streams[stream_id] = {
                'capture': cap,
                'url': rtsp_url,
                'location': location_name,
                'last_frame_time': time.time(),
                'detection_count': 0,
                'status': 'active'
            }
            
            logger.info(f"Added CCTV stream {stream_id} from {location_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding stream {stream_id}: {e}")
            return False
    
    def start_monitoring(self, rtsp_urls=None):
        """
        Start monitoring CCTV streams
        rtsp_urls: dict of {stream_id: rtsp_url} or single URL string
        """
        try:
            if isinstance(rtsp_urls, str):
                # Single stream
                success = self.add_stream('stream_1', rtsp_urls, 'Main Camera')
                if not success:
                    return False
            elif isinstance(rtsp_urls, dict):
                # Multiple streams
                for stream_id, url in rtsp_urls.items():
                    self.add_stream(stream_id, url, f'Camera {stream_id}')
            else:
                logger.error("No RTSP URLs provided")
                return False
            
            if not self.streams:
                logger.error("No active streams to monitor")
                return False
            
            self.is_monitoring = True
            logger.info(f"Starting CCTV monitoring with {len(self.streams)} streams")
            
            # Start monitoring threads for each stream
            threads = []
            for stream_id in self.streams.keys():
                thread = threading.Thread(
                    target=self._monitor_stream,
                    args=(stream_id,),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
            
            # Keep main thread alive
            try:
                while self.is_monitoring:
                    time.sleep(1)
                    self._check_stream_health()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping monitoring...")
                self.stop_monitoring()
            
            # Wait for threads to complete
            for thread in threads:
                thread.join(timeout=5)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting CCTV monitoring: {e}")
            return False
    
    def _monitor_stream(self, stream_id):
        """Monitor a single CCTV stream"""
        try:
            stream_info = self.streams[stream_id]
            cap = stream_info['capture']
            
            logger.info(f"Started monitoring stream {stream_id}")
            
            frame_count = 0
            last_detection_time = 0
            detection_cooldown = 5  # Seconds between detections for same area
            
            while self.is_monitoring and stream_info['status'] == 'active':
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read from stream {stream_id}")
                    self._attempt_reconnect(stream_id)
                    continue
                
                frame_count += 1
                stream_info['last_frame_time'] = time.time()
                
                # Process every nth frame for performance
                if frame_count % 5 != 0:
                    continue
                
                # Detect and recognize faces
                current_time = time.time()
                if current_time - last_detection_time > detection_cooldown:
                    detections = self._process_cctv_frame(frame, stream_id)
                    
                    if detections:
                        last_detection_time = current_time
                        self._handle_detections(detections, stream_id, frame)
                
                # Optional: Display frame (for debugging)
                if False:  # Set to True for debugging
                    cv2.imshow(f'CCTV Stream {stream_id}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            logger.info(f"Stopped monitoring stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Error monitoring stream {stream_id}: {e}")
        finally:
            if stream_id in self.streams:
                self.streams[stream_id]['status'] = 'stopped'
    
    def _process_cctv_frame(self, frame, stream_id):
        """Process CCTV frame for face detection and recognition"""
        try:
            detections = []
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            for face_box in faces:
                try:
                    # Extract and enhance face region
                    face_region = self.face_detector.extract_face_region(frame, face_box)
                    
                    if face_region.size == 0:
                        continue
                    
                    # Enhance low quality images (common in CCTV)
                    enhanced_face = self.face_enhancer.enhance_low_quality_image(
                        face_region, method='combined'
                    )
                    
                    # Generate encoding
                    encoding = self.face_recognizer.generate_encoding(enhanced_face)
                    
                    if encoding is not None:
                        # Find match in database
                        name, distance = self.database_manager.find_best_match(
                            encoding, self.face_recognizer
                        )
                        
                        confidence = 1.0 - distance if name != "Unknown" else 0.0
                        
                        detection = {
                            'stream_id': stream_id,
                            'face_box': face_box,
                            'name': name,
                            'confidence': confidence,
                            'timestamp': datetime.now(),
                            'face_region': enhanced_face
                        }
                        
                        detections.append(detection)
                        
                except Exception as e:
                    logger.error(f"Error processing face in stream {stream_id}: {e}")
                    continue
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing CCTV frame: {e}")
            return []
    
    def _handle_detections(self, detections, stream_id, frame):
        """Handle face detections from CCTV stream"""
        try:
            for detection in detections:
                # Log detection
                log_entry = {
                    'timestamp': detection['timestamp'].isoformat(),
                    'stream_id': stream_id,
                    'location': self.streams[stream_id]['location'],
                    'person': detection['name'],
                    'confidence': detection['confidence']
                }
                
                self.detection_log.append(log_entry)
                self.streams[stream_id]['detection_count'] += 1
                
                # Print detection info
                logger.info(
                    f"DETECTION - Stream: {stream_id}, "
                    f"Person: {detection['name']}, "
                    f"Confidence: {detection['confidence']:.2f}, "
                    f"Location: {self.streams[stream_id]['location']}"
                )
                
                # Save detection image if enabled
                if self.save_detections:
                    self._save_detection_image(detection, frame)
                
                # Generate alerts
                self._check_alerts(detection)
                
        except Exception as e:
            logger.error(f"Error handling detections: {e}")
    
    def _save_detection_image(self, detection, full_frame):
        """Save detection image to disk"""
        try:
            timestamp = detection['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"{detection['stream_id']}_{detection['name']}_{timestamp}.jpg"
            
            # Save full frame with bounding box
            annotated_frame = full_frame.copy()
            annotated_frame = ImageUtils.draw_face_box(
                annotated_frame,
                detection['face_box'],
                detection['name'],
                detection['confidence'],
                (0, 255, 0) if detection['name'] != "Unknown" else (0, 0, 255)
            )
            
            full_frame_path = self.detection_save_path / f"full_{filename}"
            ImageUtils.save_image(annotated_frame, str(full_frame_path))
            
            # Save face region
            face_path = self.detection_save_path / f"face_{filename}"
            ImageUtils.save_image(detection['face_region'], str(face_path))
            
            logger.debug(f"Saved detection images: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving detection image: {e}")
    
    def _check_alerts(self, detection):
        """Check if detection should trigger an alert"""
        try:
            should_alert = False
            alert_message = ""
            
            if detection['name'] == "Unknown" and self.unknown_person_alerts:
                should_alert = True
                alert_message = f"UNKNOWN PERSON detected in {self.streams[detection['stream_id']]['location']}"
            
            elif detection['name'] != "Unknown" and self.known_person_alerts:
                if detection['confidence'] >= self.alert_threshold:
                    should_alert = True
                    alert_message = f"{detection['name']} detected in {self.streams[detection['stream_id']]['location']}"
            
            if should_alert:
                self._send_alert(alert_message, detection)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _send_alert(self, message, detection):
        """Send alert (customize this method for your alert system)"""
        try:
            # Log alert
            logger.warning(f"ALERT: {message}")
            
            # Here you could integrate with:
            # - Email notifications
            # - SMS alerts
            # - Webhook calls
            # - Database logging
            # - Mobile app notifications
            
            # Example: Print to console with timestamp
            alert_time = detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'='*50}")
            print(f"SECURITY ALERT - {alert_time}")
            print(f"Message: {message}")
            print(f"Confidence: {detection['confidence']:.2f}")
            print(f"Stream: {detection['stream_id']}")
            print(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _attempt_reconnect(self, stream_id):
        """Attempt to reconnect to a failed stream"""
        try:
            if stream_id not in self.streams:
                return False
            
            stream_info = self.streams[stream_id]
            
            logger.info(f"Attempting to reconnect stream {stream_id}")
            
            # Close existing capture
            if stream_info['capture']:
                stream_info['capture'].release()
            
            # Wait before reconnecting
            time.sleep(5)
            
            # Try to reconnect
            cap = VideoUtils.open_video_source(stream_info['url'])
            if cap:
                stream_info['capture'] = cap
                stream_info['status'] = 'active'
                logger.info(f"Successfully reconnected stream {stream_id}")
                return True
            else:
                stream_info['status'] = 'failed'
                logger.error(f"Failed to reconnect stream {stream_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error reconnecting stream {stream_id}: {e}")
            return False
    
    def _check_stream_health(self):
        """Check health of all streams"""
        try:
            current_time = time.time()
            
            for stream_id, stream_info in self.streams.items():
                last_frame_time = stream_info['last_frame_time']
                
                # Check if stream is stale
                if current_time - last_frame_time > 30:  # 30 seconds timeout
                    logger.warning(f"Stream {stream_id} appears stale, attempting reconnect")
                    self._attempt_reconnect(stream_id)
                    
        except Exception as e:
            logger.error(f"Error checking stream health: {e}")
    
    def stop_monitoring(self):
        """Stop CCTV monitoring"""
        try:
            self.is_monitoring = False
            
            # Close all captures
            for stream_id, stream_info in self.streams.items():
                if stream_info['capture']:
                    stream_info['capture'].release()
                stream_info['status'] = 'stopped'
            
            # Save detection log
            self._save_detection_log()
            
            logger.info("CCTV monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def _save_detection_log(self):
        """Save detection log to file"""
        try:
            if self.detection_log:
                log_file = self.detection_save_path / f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_file, 'w') as f:
                    json.dump(self.detection_log, f, indent=2, default=str)
                
                logger.info(f"Saved detection log with {len(self.detection_log)} entries to {log_file}")
                
        except Exception as e:
            logger.error(f"Error saving detection log: {e}")
    
    def get_monitoring_stats(self):
        """Get monitoring statistics"""
        try:
            stats = {
                'total_streams': len(self.streams),
                'active_streams': sum(1 for s in self.streams.values() if s['status'] == 'active'),
                'total_detections': len(self.detection_log),
                'detections_per_stream': {}
            }
            
            for stream_id, stream_info in self.streams.items():
                stats['detections_per_stream'][stream_id] = stream_info['detection_count']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting monitoring stats: {e}")
            return {}

def main():
    """Main function for standalone CCTV monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCTV Face Recognition Monitor")
    parser.add_argument('--rtsp', type=str, required=True,
                       help='RTSP URL or file with multiple URLs')
    parser.add_argument('--alerts', action='store_true',
                       help='Enable unknown person alerts')
    
    args = parser.parse_args()
    
    processor = CCTVProcessor()
    
    if args.alerts:
        processor.unknown_person_alerts = True
    
    try:
        processor.start_monitoring(args.rtsp)
    except KeyboardInterrupt:
        print("\nStopping CCTV monitoring...")
        processor.stop_monitoring()

if __name__ == "__main__":
    main()