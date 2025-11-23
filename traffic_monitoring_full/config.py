import os

class Config:
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://mongo:27017/vehicle_db')
    DB_NAME = os.environ.get('DB_NAME', 'vehicle_db')
    COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'detections')
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'outputs')
    YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', '../assets/yolov8s.pt')
    CONF_THRESHOLD = float(os.environ.get('CONF_THRESHOLD', 0.3))
    IOU_THRESHOLD = float(os.environ.get('IOU_THRESHOLD', 0.4))
    TRACKER_CONFIG = os.environ.get('TRACKER_CONFIG', 'bytetrack.yaml')
    # Pixel-to-meter scale (you must calibrate for your camera) -> meters per pixel
    M_PER_PIXEL = float(os.environ.get('M_PER_PIXEL', 0.02))
    # Video fps fallback
    DEFAULT_FPS = float(os.environ.get('DEFAULT_FPS', 30.0))
    # Whether to run plate OCR (slower)
    ENABLE_PLATE_OCR = os.environ.get('ENABLE_PLATE_OCR', 'False').lower() in ('true', '1')
