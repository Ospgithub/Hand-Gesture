import os, time, urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker"
              "/hand_landmarker/float16/latest/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

FINGER_TIPS  = [8, 12, 16, 20]
FINGER_PIPS  = [6, 10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
LABELS       = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

COUNT_COLORS = {
    0: (90,  90,  90),
    1: (60, 220,  60),
    2: (0,  170, 255),
    3: (30, 140, 255),
    4: (200,  60, 200),
    5: (40,  80, 255),
}

def count_fingers(lm, is_user_right: bool) -> tuple[int, list[bool]]:
    up = []
    HYSTERESIS = 0.02

    tip_x = lm[4].x
    mcp_x = lm[2].x
    if is_user_right:
        up.append(tip_x < mcp_x - HYSTERESIS)
    else:
        up.append(tip_x > mcp_x + HYSTERESIS)

    for tip_i, pip_i in zip(FINGER_TIPS, FINGER_PIPS):
        up.append(lm[tip_i].y < lm[pip_i].y - HYSTERESIS)

    return sum(up), up


CONNS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, lm, count, up_flags, is_right_label: str):
    h, w = frame.shape[:2]
    pts  = [(int(l.x * w), int(l.y * h)) for l in lm]

    for a, b in CONNS:
        cv2.line(frame, pts[a], pts[b], (30, 190, 30), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        is_tip = i in (4, 8, 12, 16, 20)
        color  = (0, 255, 100) if is_tip else (220, 220, 220)
        radius = 7 if is_tip else 4
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)


def draw_ui(frame, hands_data):
    h, w = frame.shape[:2]

    HEADER_H = 80
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if not hands_data:
        cv2.putText(frame, "Show your hand to the camera",
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
        return

    if len(hands_data) > 1:
        total = sum(d[0] for d in hands_data)
        cv2.putText(frame, f"Total: {total}",
                    (w - 180, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 60), 2, cv2.LINE_AA)

    count, up_flags, hand_label = hands_data[0]
    color = COUNT_COLORS[count]
    label = f"{LABELS[count]}  ({count})"
    cv2.putText(frame, label, (14, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.55, color, 3, cv2.LINE_AA)
    cv2.putText(frame, hand_label + " Hand",
                (14, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 160), 1, cv2.LINE_AA)

    BAR_H  = 54
    bar_y0 = h - BAR_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y0), (w, h), (12, 12, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    col_w  = w // 5
    for i, (name, status) in enumerate(zip(FINGER_NAMES, up_flags)):
        cx = i * col_w + col_w // 2
        fc = (0, 220, 80) if status else (70, 70, 70)

        pip_y  = bar_y0 + 14
        cv2.circle(frame, (cx, pip_y), 9, fc, -1, cv2.LINE_AA)
        if status:
            cv2.circle(frame, (cx, pip_y), 9, (255, 255, 255), 1, cv2.LINE_AA)

        text_x = cx - (len(name) * 7) // 2
        cv2.putText(frame, name, (text_x, bar_y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1, cv2.LINE_AA)


class CountSmoother:
    def __init__(self, window: int = 6):
        self.buf = deque(maxlen=window)

    def update(self, count: int) -> int:
        self.buf.append(count)
        return max(set(self.buf), key=self.buf.count)


def build_detector(detection_conf=0.55, presence_conf=0.55, tracking_conf=0.5, num_hands=2):
    opts = HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=detection_conf,
        min_hand_presence_confidence=presence_conf,
        min_tracking_confidence=tracking_conf,
    )
    return HandLandmarker.create_from_options(opts)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoothers: dict[int, CountSmoother] = {}
    start_ms = int(time.time() * 1000)
    print("Webcam active — show your hand and extend fingers. Press Q to quit.")

    with build_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying..."); continue

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000) - start_ms

            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result  = detector.detect_for_video(mp_img, ts_ms)

            hands_data = []

            if result.hand_landmarks:
                for slot, (lm, handedness) in enumerate(zip(result.hand_landmarks,
                                                             result.handedness)):
                    mp_label      = handedness[0].display_name
                    is_user_right = (mp_label == "Left")
                    hand_str      = "Right" if is_user_right else "Left"

                    raw_count, up_flags = count_fingers(lm, is_user_right)

                    if slot not in smoothers:
                        smoothers[slot] = CountSmoother(window=6)
                    smooth_count = smoothers[slot].update(raw_count)

                    draw_hand(frame, lm, smooth_count, up_flags, hand_str)
                    hands_data.append((smooth_count, up_flags, hand_str))
            else:
                smoothers.clear()

            draw_ui(frame, hands_data)

            cv2.putText(frame, f"ts:{ts_ms//1000}s",
                        (frame.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

            cv2.imshow("Finger Counter  |  Task 4  [Q = quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
