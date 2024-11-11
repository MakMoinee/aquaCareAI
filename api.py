# app.py
import io
import torch
from PIL import Image
from flask import Flask, request, jsonify
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.torch_utils import select_device

app = Flask(__name__)

# device = select_device('')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./result.pt')
# model.eval()  # Set model to evaluation mode

@app.route('/detect', methods=['POST'])
def detect():
    form =  request.form
    if 'image' not in form:
        return jsonify({"error": "No image url"}), 400

    # Read the image from the request
    image = request.form['image']
    print(image)
    # Preprocess image and run YOLO model
    # img = image.convert("RGB")
    results = model(image)  # Run the model on the image

    # Parse the results (bounding boxes, labels, confidence scores)
    detections = []
    numberOfWhiteSpots =0 
    try:
        # Convert results to a Pandas DataFrame
        df = results.pandas().xyxy[0]

        # Get the number of detected objects
        numberOfWhiteSpots = df.shape[0]
        print(numberOfWhiteSpots)
        detection = {
            "whiteSpotCount": numberOfWhiteSpots
        }
        detections.append(detection)


    except Exception as e:
        print("Error:", str(e))

    if numberOfWhiteSpots>0:
        results.save(save_dir="./results/",exist_ok=True)

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run()
