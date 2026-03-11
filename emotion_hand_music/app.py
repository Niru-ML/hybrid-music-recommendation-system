from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
import webbrowser
import os

# ─── Keras / TF (optional – gracefully skip if model not found) ────────────────
try:
    from keras.models import load_model
    MODEL_AVAILABLE = os.path.exists("model.h5") and os.path.exists("labels.npy")
    if MODEL_AVAILABLE:
        emotion_model = load_model("model.h5")
        emotion_labels = np.load("labels.npy")
    else:
        emotion_model = None
        emotion_labels = None
except Exception:
    MODEL_AVAILABLE = False
    emotion_model = None
    emotion_labels = None

app = Flask(__name__)

# ─── MediaPipe Setup ───────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

holistic_processor = mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
hand_processor = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

# ─── Shared State ─────────────────────────────────────────────────────────────
shared = {
    "mode": "gesture",          # "gesture" | "emotion"
    "gesture": "No gesture",
    "emotion": "",
    "last_action": "",
    "last_action_time": 0,
    "play_state": "paused",
    "lock": threading.Lock(),
}

COOLDOWN   = 0.6
tip_ids    = [4, 8, 12, 16, 20]

# ─── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ══════════════════════════════════════════════════════════════════════════════
#  GESTURE LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def get_finger_states(lm):
    fingers = []
    # Thumb (x-axis comparison)
    fingers.append(1 if lm[4][1] < lm[3][1] else 0)
    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]][2] < lm[tip_ids[i] - 2][2] else 0)
    return fingers


def process_gesture(lm_list):
    if not lm_list:
        return "No gesture", None

    fingers = get_finger_states(lm_list)
    total   = sum(fingers)
    now     = time.time()
    gesture = "Idle"
    action  = None

    with shared["lock"]:
        last = shared["last_action_time"]

        if total == 5:
            gesture = "⏯ Play / Pause"
            if now - last > COOLDOWN:
                pyautogui.press("space")
                action = "Play/Pause toggled"
                shared["play_state"] = "playing" if shared["play_state"] == "paused" else "paused"
                shared["last_action_time"] = now

        elif total == 0:
            gesture = "🔇 Mute"
            if now - last > COOLDOWN:
                pyautogui.press("m")
                action = "Muted"
                shared["last_action_time"] = now

        elif fingers[1] == 1 and total == 1:
            x = lm_list[8][1]
            if x > 400:
                gesture = "⏩ Forward"
                if now - last > COOLDOWN:
                    pyautogui.press("right")
                    action = "Forward 5s"
                    shared["last_action_time"] = now
            elif x < 300:
                gesture = "⏪ Rewind"
                if now - last > COOLDOWN:
                    pyautogui.press("left")
                    action = "Rewind 5s"
                    shared["last_action_time"] = now

        elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
            y = lm_list[9][2]
            if y < 210:
                gesture = "🔊 Volume Up"
                if now - last > COOLDOWN:
                    pyautogui.press("up")
                    action = "Volume +"
                    shared["last_action_time"] = now
            elif y > 230:
                gesture = "🔉 Volume Down"
                if now - last > COOLDOWN:
                    pyautogui.press("down")
                    action = "Volume -"
                    shared["last_action_time"] = now

        elif fingers[0] == 1 and total == 1:
            gesture = "⏭ Next Track"
            if now - last > COOLDOWN:
                pyautogui.hotkey("ctrl", "right")
                action = "Next track"
                shared["last_action_time"] = now

        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and total == 3:
            gesture = "⏮ Prev Track"
            if now - last > COOLDOWN:
                pyautogui.hotkey("ctrl", "left")
                action = "Previous track"
                shared["last_action_time"] = now

    return gesture, action


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def process_emotion(res):
    if not MODEL_AVAILABLE or emotion_model is None:
        return None
    if not res.face_landmarks:
        return None

    lst = []
    for lm in res.face_landmarks.landmark:
        lst.append(lm.x - res.face_landmarks.landmark[1].x)
        lst.append(lm.y - res.face_landmarks.landmark[1].y)

    def hand_coords(hand_lms):
        if hand_lms:
            for lm in hand_lms.landmark:
                lst.append(lm.x - hand_lms.landmark[8].x)
                lst.append(lm.y - hand_lms.landmark[8].y)
        else:
            lst.extend([0.0] * 42)

    hand_coords(res.left_hand_landmarks)
    hand_coords(res.right_hand_landmarks)

    arr  = np.array(lst).reshape(1, -1)
    pred = emotion_labels[np.argmax(emotion_model.predict(arr, verbose=0))]
    return pred


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
def draw_hud(img, text, sub="", color=(0, 220, 180)):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (16, 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def gen_gesture_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        img    = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res    = hand_processor.process(rgb)
        rgb.flags.writeable = True
        img    = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lm_list = []
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hl, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2),
                )
            myHand = res.multi_hand_landmarks[0]
            h, w, _ = img.shape
            for i, lm in enumerate(myHand.landmark):
                lm_list.append([i, int(lm.x * w), int(lm.y * h)])

        gesture, action = process_gesture(lm_list)

        with shared["lock"]:
            shared["gesture"] = gesture
            if action:
                shared["last_action"] = action

        sub = f"Action: {shared['last_action']}" if shared["last_action"] else ""
        draw_hud(img, gesture, sub)

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


def gen_emotion_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic_processor.process(rgb)
        rgb.flags.writeable = True
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Draw face mesh
        if res.face_landmarks:
            mp_drawing.draw_landmarks(
                img, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=-1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 100, 200), thickness=1),
            )
        mp_drawing.draw_landmarks(img, res.left_hand_landmarks,  mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        emotion = process_emotion(res)
        if emotion:
            with shared["lock"]:
                shared["emotion"] = emotion

        emo_display = shared["emotion"] or ("Model not loaded" if not MODEL_AVAILABLE else "Detecting…")
        draw_hud(img, f"Emotion: {emo_display}", "", color=(255, 160, 60))

        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/gesture")
def gesture_page():
    return render_template("gesture.html")


@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html")


@app.route("/video/gesture")
def video_gesture():
    return Response(gen_gesture_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/emotion")
def video_emotion():
    return Response(gen_emotion_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    with shared["lock"]:
        return jsonify({
            "gesture":     shared["gesture"],
            "emotion":     shared["emotion"],
            "last_action": shared["last_action"],
            "play_state":  shared["play_state"],
            "model_ready": MODEL_AVAILABLE,
        })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data   = request.get_json()
    lang   = data.get("lang", "english")
    singer = data.get("singer", "")
    with shared["lock"]:
        emotion = shared["emotion"]
    if not emotion:
        return jsonify({"ok": False, "msg": "No emotion detected yet"})
    query = f"{lang} {emotion} song {singer}".strip()
    url   = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    webbrowser.open(url)
    return jsonify({"ok": True, "url": url, "emotion": emotion})


@app.route("/api/set_mode", methods=["POST"])
def api_set_mode():
    data = request.get_json()
    mode = data.get("mode", "gesture")
    with shared["lock"]:
        shared["mode"] = mode
    return jsonify({"ok": True, "mode": mode})


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
