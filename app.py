import os, time, threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, request

from Model import (
    MODEL_PATH, FINGER_NAMES, LABELS, COUNT_COLORS,
    count_fingers, draw_hand, draw_ui, build_detector, CountSmoother
)

app = Flask(__name__)

_state = {
    "count":       0,
    "label":       "Zero",
    "fingers":     [False] * 5,
    "hand":        "—",
    "fps":         0.0,
    "frame":       None,
    "lock":        threading.Lock(),
    "det_conf":    0.55,
    "track_conf":  0.50,
    "smooth_win":  6,
    "restart":     False,
}


def _camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoothers: dict[int, CountSmoother] = {}
    start_ms   = int(time.time() * 1000)
    fps_buf    = deque(maxlen=30)
    prev_t     = time.perf_counter()

    detector = build_detector(
        detection_conf=_state["det_conf"],
        tracking_conf=_state["track_conf"],
    )

    while True:
        if _state["restart"]:
            detector.close()
            smoothers.clear()
            detector = build_detector(
                detection_conf=_state["det_conf"],
                tracking_conf=_state["track_conf"],
            )
            _state["restart"] = False

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.033)
            continue

        now_t  = time.perf_counter()
        fps_buf.append(1.0 / max(now_t - prev_t, 1e-6))
        prev_t = now_t
        fps    = sum(fps_buf) / len(fps_buf)

        frame  = cv2.flip(frame, 1)
        ts_ms  = int(time.time() * 1000) - start_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect_for_video(mp_img, ts_ms)

        hands_data  = []
        primary_count  = 0
        primary_flags  = [False] * 5
        primary_hand   = "—"

        if result.hand_landmarks:
            for slot, (lm, handedness) in enumerate(zip(result.hand_landmarks,
                                                         result.handedness)):
                mp_label      = handedness[0].display_name
                is_user_right = (mp_label == "Left")
                hand_str      = "Right" if is_user_right else "Left"
                raw_count, up_flags = count_fingers(lm, is_user_right)

                win = _state["smooth_win"]
                if slot not in smoothers or smoothers[slot].buf.maxlen != win:
                    smoothers[slot] = CountSmoother(window=win)
                smooth_count = smoothers[slot].update(raw_count)

                draw_hand(frame, lm, smooth_count, up_flags, hand_str)
                hands_data.append((smooth_count, up_flags, hand_str))

                if slot == 0:
                    primary_count = smooth_count
                    primary_flags = up_flags
                    primary_hand  = hand_str
        else:
            smoothers.clear()

        draw_ui(frame, hands_data)

        cv2.putText(frame, f"{fps:.1f} FPS",
                    (frame.shape[1] - 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 200, 80), 1, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

        with _state["lock"]:
            _state["frame"]   = jpeg.tobytes()
            _state["count"]   = primary_count
            _state["label"]   = LABELS[primary_count]
            _state["fingers"] = primary_flags
            _state["hand"]    = primary_hand
            _state["fps"]     = round(fps, 1)

    cap.release()


threading.Thread(target=_camera_thread, daemon=True).start()


def _gen_frames():
    while True:
        with _state["lock"]:
            frame = _state["frame"]
        if frame is None:
            time.sleep(0.033)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with _state["lock"]:
        return jsonify({
            "count":   _state["count"],
            "label":   _state["label"],
            "fingers": _state["fingers"],
            "hand":    _state["hand"],
            "fps":     _state["fps"],
        })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    import os, signal
    os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"ok": True, "message": "Server shutting down…"})


@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json(force=True)
    changed = False
    if "det_conf" in data:
        _state["det_conf"]   = float(data["det_conf"])
        changed = True
    if "track_conf" in data:
        _state["track_conf"] = float(data["track_conf"])
        changed = True
    if "smooth_win" in data:
        _state["smooth_win"] = int(data["smooth_win"])
    if changed:
        _state["restart"] = True
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Starting Hand Gesture Recognition server …")
    print("Open  http://127.0.0.1:5000  in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
