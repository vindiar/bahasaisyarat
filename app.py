import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Pastikan folder static ada untuk menyimpan file upload
os.makedirs("static", exist_ok=True)

# Load ONNX model menggunakan OpenCV DNN
net = cv2.dnn.readNetFromONNX("model/my_model.onnx")

# Daftar nama kelas SIBI (A-Z)
CLASSES = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

def run_inference(img):
    # Preprocessing image untuk YOLOv8 (imgsz=320)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    # Format output YOLOv8: [1, 30, 2100] (di mana 30 = 4 box koordinat + 26 class probabilities)
    # Kita ambil deteksi dengan skor tertinggi
    predictions = outputs[0]  # shape: (30, 2100)
    predictions = np.transpose(predictions)  # shape: (2100, 30)
    
    best_class_id = -1
    best_score = 0.0
    best_box = None
    
    for pred in predictions:
        scores = pred[4:]
        class_id = np.argmax(scores)
        score = scores[class_id]
        if score > 0.4 and score > best_score:  # threshold 0.4
            best_score = score
            best_class_id = class_id
            best_box = pred[0:4]
            
    # Gambar bounding box jika terdeteksi
    h, w = img.shape[:2]
    pred_label = "No detection"
    
    if best_class_id != -1 and best_box is not None:
        pred_label = CLASSES[best_class_id]
        # YOLOv8 format: [x_center, y_center, width, height] normalized to 320x320
        # Kita sesuaikan ke ukuran gambar asli
        x_center, y_center, box_w, box_h = best_box
        x_factor = w / 320.0
        y_factor = h / 320.0
        
        left = int((x_center - box_w / 2) * x_factor)
        top = int((y_center - box_h / 2) * y_factor)
        width = int(box_w * x_factor)
        height = int(box_h * y_factor)
        
        # Gambar box dan label di image
        cv2.rectangle(img, (left, top), (left + width, top + height), (180, 200, 255), 3)
        cv2.putText(img, f"{pred_label} ({best_score:.2f})", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 200, 255), 2)
                    
    return img, pred_label

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

    img = cv2.imread(filepath)
    annotated_img, pred = run_inference(img)
    cv2.imwrite("static/result.jpg", annotated_img)

    return jsonify({
        "prediction": pred,
        "image": "/static/result.jpg"
    })

# ======================
# WEBCAM FRAME DETECTION
# ======================
@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    data = request.json['image']

    # Decode base64
    encoded = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated_frame, pred = run_inference(img)

    # Encode kembali ke base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": img_base64,
        "prediction": pred
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)