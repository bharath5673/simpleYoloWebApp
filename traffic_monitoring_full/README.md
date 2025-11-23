 # Vehicle Traffic Monitoring — Flask + MongoDB (All-in-one)

 ## Quick start (local)

 1. Create a Python virtualenv and activate it.
e.g. `python3 -m venv venv && source venv/bin/activate`

 2. Install packages:
`pip install -r requirements.txt`

 3. Ensure you have a YOLOv8 model at the path configured in `config.py` (default: `../assets/yolov8s.pt`).

 4. Start MongoDB (optional) or use the included docker-compose:
    `docker-compose up -d mongo`

 5. Run the Flask app:
    `python3 app.py`

 6. Open your browser at `http://127.0.0.1:5000`.
    - Upload a video
    - Click Start Detection
    - Watch live processed frames and find the saved output in `outputs/`.

 ## Notes
 - Calibrate `M_PER_PIXEL` inside `config.py` for real-world speed accuracy.
 - OCR is optional and can be enabled via env `ENABLE_PLATE_OCR=true`.
 - For GPU acceleration, install the correct `torch` version for your CUDA.
 - This is a minimal prototype; consider batching DB writes, handling large videos, and adding auth for production.
