import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Gesture label (forward, left, right, stop, up, down, slow, rotate_left, rotate_right): ")


with open("hand_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                row = []
                for lm in handLms.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                row.append(label)
                writer.writerow(row)
                print("Saved")

                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Collecting Data", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
