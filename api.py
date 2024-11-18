# app.py
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
import os
import pathlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
from contextlib import redirect_stdout

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress torch hub's internal messages
torch.hub._verbose = False

# Modify pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app, origins=["http://localhost:8443"])

# Load the model without showing cache loading and model summary
with redirect_stdout(io.StringIO()):  # Suppress YOLOv5 stdout messages
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./result.pt', verbose=False)
    model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='./human.pt', verbose=False)

executor = ThreadPoolExecutor()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.form:
        logger.info("Received request without image URL in /detect")
        return jsonify({"error": "No image url"}), 400

    # Log when an image URL is received for detection
    image = request.form['image']
    logger.info("Received image URL for detection in /detect")

    # Indicate detection process started
    detections = [{"detection": "in progress"}]
    executor.submit(do_object_detection, image)

    return jsonify({"detections": detections})

def do_object_detection(image):
    
    try:
        results2 = model2(image)
        df2 = results2.pandas().xyxy[0]
        check = df2.shape[0]
        
        if check > 0:
            logger.info(f"Detected {check} humans and other stuff in picture.")
        else:
            try:
                results = model(image)
                df = results.pandas().xyxy[0]
                numberOfWhiteSpots = df.shape[0]
                if numberOfWhiteSpots > 0:
                    results.save(save_dir="./results/", exist_ok=True)
                    logger.info(f"Detected {numberOfWhiteSpots} white spots and saved results.")
                else:
                    logger.info("No white spots detected.")
            except Exception as e:
                logger.error(f"Error during detection: {e}")
    except Exception as e:
        logger.error(f"Error during detection: {e}")

@app.route('/show_results', methods=['GET'])
def show_results():
    results_dir = "./results"
    images = [filename for filename in os.listdir(results_dir)
              if filename.endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(results_dir) else []
    
    logger.info("Accessed /show_results to list saved images.")
    return jsonify({"images": images})

@app.route('/display_image/<filename>')
def display_image(filename):
    try:
        img_path = os.path.join("./results", filename)
        logger.info(f"Displaying image: {filename} from /display_image")
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Error displaying image {filename}: {e}")
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    # Set Flask to only log critical errors
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    logger.info("Starting API server...")
    app.run()
