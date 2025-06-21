from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from emotions import emotion_dict  # Function to map predictions to emotion labels

app = Flask(__name__)
app.secret_key = b"\xa3\xc8\xb7\xb5\xc1E\x16\xd4\x05\x86\xe4\xa1h\x8b\xbd\xbf\t?\xd2\xe4v\x91\x82Z"
#emotion_log = []  # To store detected emotions with timestamp
#MAX_LOG_LENGTH = 500

users_file = os.path.join(os.path.dirname(__file__), "data", "users.json")

# Load pre-trained emotion model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input

# Recreate the model architecture
model = Sequential()

# Explicitly define input as a separate layer
model.add(Input(shape=(48, 48, 1)))

# Now add the Conv2D layers without input_shape
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes for emotion detection
# Load weights from model.h5
model.load_weights("model.h5")

# Print model summary
model.summary()



# Load face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read users from JSON file
def read_users():
    if os.path.exists(users_file):
        with open(users_file, "r") as file:
            return json.load(file)
    return []

# Write users to JSON file
def write_users(users):
    with open(users_file, "w") as file:
        json.dump(users, file, indent=4)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    users = read_users()
    user = next((user for user in users if user["username"] == username and user["password"] == password), None)
    if user:
        session["username"] = user["username"]  # Set session
        return redirect(url_for("home"))
    else:
        return jsonify({"message": "Invalid credentials", "is_user": False}), 400

@app.route("/home")
def home():
    if "username" in session:
        return render_template("home.html")
    else:
        return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/emotion-history")
def emotion_history():
    return render_template("emotion_history.html", emotions=emotion_log)


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect('/')
@app.route("/chat-history", methods=["GET"])
def chat_history():
    # Replace with your logic to fetch past chat messages
    messages = [
        {"sender": "User", "text": "Hello"},
        {"sender": "Bot", "text": "Hi, how can I help you?"},
        # ... load from database/file/session if needed
    ]
    return render_template("chat_history.html", messages=messages)




@app.route('/clear-history')
def clear_history():
    # Clear session/chat history here
    session.pop('chat_history', None)
    return redirect('/')

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    users = read_users()
    if any(user["username"] == username for user in users):
        return jsonify({"message": "Username already exists"}), 400
    users.append({"username": username, "password": password})
    write_users(users)
    return jsonify({"message": "Registration successful"})

from flask import Response

from datetime import datetime

# ── NEW ─────────────────────────────────────────────────────────────────────────
emotion_log = []          # Will hold {"emotion": str, "time": str} dictionaries
MAX_LOG_LENGTH = 500      # Optional: keep only the most recent N entries
# ────────────────────────────────────────────────────────────────────────────────

def generate_frames():
    """
    Capture webcam frames, detect faces, predict emotions, store detections
    in emotion_log, and stream the annotated video.
    """
    camera = cv2.VideoCapture(0)        # Open webcam

    while True:
        success, frame = camera.read()
        if not success:
            print("[ERROR] Could not read frame from webcam")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("[INFO] No face detected")

        for (x, y, w, h) in faces:
            # --- Pre‑process ROI for the model ---
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=[0, -1])

            # --- Predict emotion ---
            prediction    = model.predict(roi)
            emotion_label = emotion_dict[int(np.argmax(prediction))]
            timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # --- Store in global log (with size cap) ---
            emotion_log.append({"emotion": emotion_label, "time": timestamp})
            if len(emotion_log) > MAX_LOG_LENGTH:
                emotion_log.pop(0)  # Discard oldest entry

            print(f"[INFO] {timestamp} – Detected Emotion: {emotion_label}")

            # --- Annotate frame ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- Convert annotated frame to JPEG & yield to browser ---
        _, buffer      = cv2.imencode(".jpg", frame)
        frame_bytes    = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    camera.release()


@app.route("/real_time_emotion")
def real_time_emotion():
    """Return real-time emotion detection video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
