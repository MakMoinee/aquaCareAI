# app.py
import io
import torch
from PIL import Image
from quart import Quart, jsonify, request
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

app = Quart(__name__)

# Load the YOLOv5 model
device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device)  # Replace 'yolov5s.pt' with your model path
model.eval()  # Set model to evaluation mode

@app.route('/detect', methods=['POST'])
async def detect():
    if not request.files.get("image"):
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image from the request
    image_file = await request.files["image"].read()
    image = Image.open(io.BytesIO(image_file))

    # Preprocess image and run YOLO model
    img = image.convert("RGB")
    results = model(img)  # Run the model on the image

    # Parse the results (bounding boxes, labels, confidence scores)
    detections = []
    for *box, conf, cls in results.xyxy[0]:  # xyxy format
        detection = {
            "bbox": [round(x, 2) for x in box],  # Bounding box coordinates
            "confidence": round(float(conf), 2),
            "class": int(cls)
        }
        detections.append(detection)

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run()
