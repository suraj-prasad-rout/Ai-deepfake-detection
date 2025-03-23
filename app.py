from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import os
import tempfile
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ✅ Load Model (Ensure it's properly loaded)
MODEL_PATH = r"E:\AI dfde frontend\deepfake_detection_model.h5"  #aded by suraj made changes in pathas per you local model file destination

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    model = None  # Prevent API from crashing if the model fails to load

# ✅ Function to Extract Frames from Video


def extract_frames(video_path, target_size=(224, 224), max_frames=50):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Cannot open video file.")
        return []

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    frame_interval = max(1, total_frames // max_frames)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frame = frame / 255.0  # Normalize
            frames.append(frame)

            if len(frames) >= max_frames:
                break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()  # ✅ Ensure OpenCV releases file handles
    return np.array(frames)


@app.route('/')
def home():
    return "✅ Deepfake Detection API is Running!"


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename.lower()

    try:
        # ✅ Handle Image File
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image) / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            prediction = model.predict(image)
            result = "Real" if prediction[0][0] < 0.5 else "Fake"

            return jsonify({
                "file_type": "image",
                "prediction": result,
                "confidence": float(prediction[0][0])
            })

        # ✅ Handle Video File
        elif filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name

            # ✅ Extract Frames
            frames = extract_frames(temp_file_path)

            # ✅ Ensure the file is completely released before deletion
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)  # ✅ Delete temp file safely
                except Exception as e:
                    print(f"⚠️ Warning: Could not delete temp file: {e}")

            if len(frames) == 0:
                return jsonify({"error": "No valid frames extracted from video"}), 400

            # ✅ Predict on Extracted Frames & Take Majority Decision
            predictions = model.predict(frames)
            avg_prediction = np.mean(predictions)
            result = "Real" if avg_prediction < 0.5 else "Fake"

            return jsonify({
                "file_type": "video",
                "prediction": result,
                "confidence": float(avg_prediction),
                "total_frames_analyzed": len(frames)
            })

        else:
            return jsonify({"error": "Unsupported file format"}), 400

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
