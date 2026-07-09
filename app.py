from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = Flask(__name__)

# load model
model = YOLO("model/my_model.pt")

@app.route('/')
def home():
    return render_template("index.html")


# ======================
# UPLOAD GAMBAR
# ======================
@app.route('/predict-image', methods=['POST'])
def predict_image():
    file = request.files['image']
    filepath = "static/upload.jpg"
    file.save(filepath)

    results = model(filepath)

    # save hasil dengan bounding box
    results[0].save(filename="static/result.jpg")

    pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"

    return jsonify({
        "prediction": pred,
        "image": "/static/result.jpg"
    })


@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    data = request.json['image']

    # decode base64
    encoded = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 🔥 INI HARUS DI DALAM FUNCTION
    results = model(img, imgsz=320)

    # gambar bounding box
    annotated_frame = results[0].plot()

    # ambil label
    pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"

    # encode ke base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": img_base64,
        "prediction": pred
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)