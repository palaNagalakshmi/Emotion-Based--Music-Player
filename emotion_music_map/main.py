import os
import cv2
import time
import random
import threading
import pygame
import numpy as np
from flask import Flask, render_template, Response, jsonify
from deepface import DeepFace

app = Flask(__name__)
pygame.mixer.init()

# Music path mapping
MUSIC_FOLDER = "music"
EMOTION_MAP = {
    "angry": os.path.join(MUSIC_FOLDER, "angry"),
    "disgust": os.path.join(MUSIC_FOLDER, "disgust"),
    "fear": os.path.join(MUSIC_FOLDER, "fear"),
    "happy": os.path.join(MUSIC_FOLDER, "happy"),
    "neutral": os.path.join(MUSIC_FOLDER, "neutral"),
    "sad": os.path.join(MUSIC_FOLDER, "sad"),
    "surprise": os.path.join(MUSIC_FOLDER, "surprise")
}

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Globals
current_emotion = "Detecting..."
current_song = "None"
last_emotion = None
cap = cv2.VideoCapture(0)

def get_song(emotion):
    folder = EMOTION_MAP.get(emotion, "")
    if folder and os.path.exists(folder):
        songs = [os.path.join(folder, song) for song in os.listdir(folder) if song.endswith(".mp3")]
        if songs:
            return random.choice(songs)
    return None

def detect_emotion_loop():
    global current_emotion, current_song, last_emotion

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for faster detection and UI display
        frame_small = cv2.resize(frame, (480, 360))

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            try:
                result = DeepFace.analyze(frame_small, actions=["emotion"], enforce_detection=False)
                dominant_emotion = result[0]["dominant_emotion"]
                current_emotion = dominant_emotion

                if dominant_emotion != last_emotion:
                    song = get_song(dominant_emotion)
                    if song:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(song)
                        pygame.mixer.music.play()
                        current_song = os.path.basename(song)
                        last_emotion = dominant_emotion
            except Exception as e:
                print("Emotion detection error:", e)
                current_emotion = "Error"
                current_song = "None"
        else:
            current_emotion = "No Face Detected"
            current_song = "None"

        time.sleep(3)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.resize(frame, (480, 360))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
def detect_emotion():
    return jsonify({
        "emotion": current_emotion,
        "song": current_song
    })

@app.route('/stop_music')
def stop_music():
    pygame.mixer.music.stop()
    return "Music stopped", 200

if __name__ == "__main__":
    emotion_thread = threading.Thread(target=detect_emotion_loop, daemon=True)
    emotion_thread.start()
    app.run(debug=True)
