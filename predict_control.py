import cv2
import mediapipe as mp
import joblib
import math

model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

speed_mode = "NORMAL"

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    action = "No Hand"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            data = []
            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)

            pred = model.predict([data])[0]

            # Speed control
            if pred == "slow":
                speed_mode = "SLOW"
            elif pred == "fast":
                speed_mode = "FAST"

            # Movement control
            if pred == "left":
                action = f"Turn Left ({speed_mode})"
            elif pred == "right":
                action = f"Turn Right ({speed_mode})"
            elif pred == "forward":
                action = f"Forward ({speed_mode})"
            elif pred == "backward":
                action = f"Backward ({speed_mode})"
            elif pred == "up":
                action = f"Up ({speed_mode})"
            elif pred == "down":
                action = f"Down ({speed_mode})"
            elif pred == "rotate_left":
                action = "Rotate Left"
            elif pred == "rotate_right":
                action = "Rotate Right"
            elif pred == "stop":
                action = "Emergency Stop"

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(img, action, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(img, f"Speed: {speed_mode}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Drone Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
