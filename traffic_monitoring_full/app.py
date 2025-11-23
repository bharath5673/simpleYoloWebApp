import os, time
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import cv2

from config import Config
from detector import Detector

UPLOAD_FOLDER = Config.UPLOAD_FOLDER
OUTPUT_FOLDER = Config.OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mongo
client = MongoClient(Config.MONGO_URI)
db = client[Config.DB_NAME]
collection = db[Config.COLLECTION_NAME]

# Detector instance
detector = Detector(collection)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'message': 'Uploaded', 'path': filepath}), 201

@app.route('/start', methods=['POST'])
def start_processing():
    data = request.json or {}
    video_path = data.get('video_path')
    output_path = data.get('output_path') or os.path.join(OUTPUT_FOLDER, 'output.mp4')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'video_path missing or not found'}), 400
    started = detector.start(video_path, output_path)
    return jsonify({'started': started, 'output_path': output_path}), 200

@app.route('/stop', methods=['POST'])
def stop_processing():
    detector.stop()
    return jsonify({'stopped': True}), 200

@app.route('/detections', methods=['GET'])
def get_detections():
    q = {}
    object_id = request.args.get('object_id')
    cls = request.args.get('class')
    limit = int(request.args.get('limit', 100))
    if object_id:
        try:
            q['object_id'] = int(object_id)
        except Exception:
            pass
    if cls:
        q['class'] = cls
    cursor = collection.find(q).sort('timestamp', -1).limit(limit)
    docs = []
    for d in cursor:
        d['_id'] = str(d['_id'])
        docs.append(d)
    return jsonify(docs)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if detector.latest_frame is not None:
                frame_bytes = detector.latest_frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
