import argparse
import sys
import logging
from pathlib import Path

from training.train_encodings import FaceEncodingTrainer
from models.database_manager import DatabaseManager
from real_time_recognition import RealTimeFaceRecognition
from batch_processing import BatchProcessor
from cctv_integration import CCTVProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    from config.config import DATA_DIR, KNOWN_FACES_DIR, ENCODINGS_DIR, TEST_IMAGES_DIR
    
    directories = [DATA_DIR, KNOWN_FACES_DIR, ENCODINGS_DIR, TEST_IMAGES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def train_command(args):
    """Handle training command"""
    trainer = FaceEncodingTrainer()
    
    if args.person and args.image:
        # Add single face
        success = trainer.add_single_face(args.image, args.person)
        if success:
            print(f"Successfully added face for {args.person}")
        else:
            print(f"Failed to add face for {args.person}")
    elif args.person:
        # Retrain specific person
        success = trainer.retrain_person(args.person)
        if success:
            print(f"Successfully retrained {args.person}")
        else:
            print(f"Failed to retrain {args.person}")
    else:
        # Train all faces
        success = trainer.train_from_directory(args.directory)
        if success:
            print("Training completed successfully!")
        else:
            print("Training failed!")

def recognize_command(args):
    """Handle recognition command"""
    if args.realtime:
        # Real-time recognition
        recognizer = RealTimeFaceRecognition()
        recognizer.start_recognition(source=args.source)
    elif args.batch:
        # Batch processing
        processor = BatchProcessor()
        processor.process_directory(args.input, args.output)
    elif args.cctv:
        # CCTV processing
        cctv_processor = CCTVProcessor()
        cctv_processor.start_monitoring(args.source)
    else:
        print("Please specify recognition mode: --realtime, --batch, or --cctv")

def database_command(args):
    """Handle database commands"""
    db_manager = DatabaseManager()
    
    if args.list:
        # List all people in database
        people = db_manager.get_face_names()
        if people:
            print(f"Database contains {len(people)} people:")
            for person in people:
                count = db_manager.get_face_count(person)
                print(f"  {person}: {count} encodings")
        else:
            print("Database is empty")
    
    elif args.remove:
        # Remove person from database
        success = db_manager.remove_face(args.remove)
        if success:
            db_manager.save_encodings()
            print(f"Removed {args.remove} from database")
        else:
            print(f"Failed to remove {args.remove}")
    
    elif args.info:
        # Show database info
        info = db_manager.export_database_info()
        print(f"Database Statistics:")
        print(f"  Total People: {info['total_people']}")
        print(f"  Total Encodings: {info['total_encodings']}")
        print(f"  Database File: {db_manager.encodings_file}")

def main():
    """Main application entry point"""
    setup_directories()
    
    parser = argparse.ArgumentParser(description="Advanced Facial Recognition System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Train face encodings')
    train_parser.add_argument('--directory', type=str, help='Directory containing face images')
    train_parser.add_argument('--person', type=str, help='Person name for single face training')
    train_parser.add_argument('--image', type=str, help='Image path for single face training')
    
    # Recognition commands
    recognize_parser = subparsers.add_parser('recognize', help='Face recognition')
    recognize_parser.add_argument('--realtime', action='store_true', help='Real-time recognition')
    recognize_parser.add_argument('--batch', action='store_true', help='Batch processing')
    recognize_parser.add_argument('--cctv', action='store_true', help='CCTV monitoring')
    recognize_parser.add_argument('--source', type=str, default=0, help='Video source (webcam/file/RTSP)')
    recognize_parser.add_argument('--input', type=str, help='Input directory for batch processing')
    recognize_parser.add_argument('--output', type=str, help='Output directory for batch processing')
    
    # Database commands
    db_parser = subparsers.add_parser('database', help='Database management')
    db_parser.add_argument('--list', action='store_true', help='List all people in database')
    db_parser.add_argument('--remove', type=str, help='Remove person from database')
    db_parser.add_argument('--info', action='store_true', help='Show database information')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'recognize':
        recognize_command(args)
    elif args.command == 'database':
        database_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
