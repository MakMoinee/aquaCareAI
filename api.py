# app.py
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8443"]) 

# device = select_device('')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./result.pt')
executor = ThreadPoolExecutor()

@app.route('/detect', methods=['POST'])
def detect():
    form =  request.form
    if 'image' not in form:
        return jsonify({"error": "No image url"}), 400

    # Read the image from the request
    image = request.form['image']
    print(image)
    detections = []
    detection = {
        "detection": "in progress"
    }
    detections.append(detection)
    executor.submit(do_object_detection, image)

    return jsonify({"detections": detections})

def do_object_detection(image):
    results = model(image)
    detections = []
    numberOfWhiteSpots =0 
    try:
        # Convert results to a Pandas DataFrame
        df = results.pandas().xyxy[0]

        # Get the number of detected objects
        numberOfWhiteSpots = df.shape[0]
        print(numberOfWhiteSpots)


    except Exception as e:
        print("Error:", str(e))

    if numberOfWhiteSpots>0:
        results.save(save_dir="./results/",exist_ok=True)



@app.route('/show_results', methods=['GET'])
def show_results():
    images = []
    results_dir = "./results"
    
    # List all images in the results folder
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                images.append(filename)
                
    return jsonify({"images": images})

@app.route('/display_image/<filename>')
def display_image(filename):
    try:
        img_path = os.path.join("./results", filename)
        return send_file(img_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    app.run()
