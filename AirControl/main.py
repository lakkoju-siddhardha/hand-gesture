import cv2
import mediapipe as mp
import pyautogui
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================
# MediaPipe Hand Landmarker
# =============================

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


# =============================
# Webcam
# =============================

cap = cv2.VideoCapture(0)

timestamp = 0

# Starting hand position
anchor_y = None

# Movement required to trigger page change
MOVEMENT_THRESHOLD = 0.12

# Prevent repeated page changes
COOLDOWN = 0.8

last_action_time = 0


# =============================
# Main Loop
# =============================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not access camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 33

    # Detect hand
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # =============================
    # Hand Movement Detection
    # =============================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        # Wrist Y coordinate
        current_y = hand[0].y

        # Set starting position
        if anchor_y is None:
            anchor_y = current_y

        # Calculate movement
        movement = current_y - anchor_y

        current_time = time.time()


        # =============================
        # MOVE UP → PREVIOUS PAGE
        # =============================

        if (
            movement < -MOVEMENT_THRESHOLD
            and current_time - last_action_time > COOLDOWN
        ):

            print("⬆️ PREVIOUS PAGE")

            pyautogui.press("pageup")

            last_action_time = current_time

            # Reset starting position
            anchor_y = current_y


        # =============================
        # MOVE DOWN → NEXT PAGE
        # =============================

        elif (
            movement > MOVEMENT_THRESHOLD
            and current_time - last_action_time > COOLDOWN
        ):

            print("⬇️ NEXT PAGE")

            pyautogui.press("pagedown")

            last_action_time = current_time

            # Reset starting position
            anchor_y = current_y


        # =============================
        # Draw Hand
        # =============================

        h, w, _ = frame.shape

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    else:

        # Reset when hand disappears
        anchor_y = None


    # =============================
    # Display Camera
    # =============================

    cv2.imshow(
        "AirControl - PDF Page Control",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =============================
# Cleanup
# =============================

cap.release()
cv2.destroyAllWindows()
landmarker.close()