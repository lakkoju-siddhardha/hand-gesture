import cv2
import mediapipe as mp
import pyautogui
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# PyAutoGUI
# ==========================================

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

SCREEN_W, SCREEN_H = pyautogui.size()


# ==========================================
# MediaPipe Hand Landmarker
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)


# ==========================================
# Camera
# ==========================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ==========================================
# Mouse settings
# ==========================================

FRAME_REDUCTION = 80
SMOOTHING = 5

previous_x = 0
previous_y = 0

clicking = False

PINCH_THRESHOLD = 0.045


# ==========================================
# Swipe settings
# ==========================================

SWIPE_DISTANCE = 0.14
SWIPE_TIME = 0.45
SWIPE_COOLDOWN = 0.8

start_y = None
start_time = None
last_swipe_time = 0

gesture_text = "READY"


# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Camera unavailable")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int(time.monotonic() * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    h, w, _ = frame.shape

    gesture_text = "SHOW HAND"


    # ==========================================
    # Hand detected
    # ==========================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index_tip = hand[8]
        thumb_tip = hand[4]
        wrist = hand[0]
        middle_mcp = hand[9]


        # ======================================
        # 1. INDEX FINGER → MOUSE MOVEMENT
        # ======================================

        target_x = int(
            (index_tip.x * w - FRAME_REDUCTION)
            * SCREEN_W
            / (w - 2 * FRAME_REDUCTION)
        )

        target_y = int(
            (index_tip.y * h - FRAME_REDUCTION)
            * SCREEN_H
            / (h - 2 * FRAME_REDUCTION)
        )

        target_x = max(0, min(SCREEN_W - 1, target_x))
        target_y = max(0, min(SCREEN_H - 1, target_y))

        smooth_x = previous_x + (
            target_x - previous_x
        ) / SMOOTHING

        smooth_y = previous_y + (
            target_y - previous_y
        ) / SMOOTHING

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y)
        )

        previous_x = smooth_x
        previous_y = smooth_y


        # ======================================
        # 2. PINCH → LEFT CLICK
        # ======================================

        distance = math.hypot(
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y
        )

        if distance < PINCH_THRESHOLD:

            gesture_text = "PINCH / CLICK"

            # Click once when pinch begins
            if not clicking:
                pyautogui.click()
                clicking = True

        else:
            clicking = False


        # ======================================
        # 3. SWIPE → PDF PAGE NAVIGATION
        # ======================================

        current_time = time.monotonic()

        current_y = (
            wrist.y + middle_mcp.y
        ) / 2

        # Don't interpret the same pinch motion
        # as a swipe.
        if not clicking:

            if start_y is None:
                start_y = current_y
                start_time = current_time

            movement = current_y - start_y
            elapsed = current_time - start_time

            if (
                elapsed <= SWIPE_TIME
                and current_time - last_swipe_time
                > SWIPE_COOLDOWN
            ):

                if movement < -SWIPE_DISTANCE:

                    print("SWIPE UP: PREVIOUS PAGE")

                    pyautogui.press("pageup")

                    gesture_text = "PREVIOUS PAGE"

                    last_swipe_time = current_time
                    start_y = None
                    start_time = None

                elif movement > SWIPE_DISTANCE:

                    print("SWIPE DOWN: NEXT PAGE")

                    pyautogui.press("pagedown")

                    gesture_text = "NEXT PAGE"

                    last_swipe_time = current_time
                    start_y = None
                    start_time = None

            elif elapsed > SWIPE_TIME:
                start_y = current_y
                start_time = current_time

        else:
            start_y = None
            start_time = None


        # ======================================
        # Draw index and thumb tips
        # ======================================

        for point in [index_tip, thumb_tip]:

            px = int(point.x * w)
            py = int(point.y * h)

            cv2.circle(
                frame,
                (px, py),
                8,
                (0, 255, 0),
                -1
            )

    else:

        clicking = False
        start_y = None
        start_time = None

        previous_x = 0
        previous_y = 0


    # ==========================================
    # Display status
    # ==========================================

    cv2.putText(
        frame,
        gesture_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("AirControl - Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()